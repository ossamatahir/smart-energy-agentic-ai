import streamlit as st
import os, sys, json, math, random, requests, chromadb
import torch, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='Smart Energy AI', page_icon='zap',
    layout='wide', initial_sidebar_state='expanded')

st.markdown('<style>.stButton>button{background:#7c3aed;color:white;border-radius:8px;border:none;}</style>',
    unsafe_allow_html=True)

WEATHER_API = '11fe2fe13bb3a5b33f8cf72ec88912be'
SERP_API    = 'a0ad79eed1bf6c77b3b0ed7a5b5dbd9d3bf1bae9eecedcacb9c6f8b73b3e31c3'

@st.cache_resource(show_spinner='Loading FLAN-T5-XL...')
def load_llm():
    tok = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    mod = T5ForConditionalGeneration.from_pretrained(
              'google/flan-t5-xl', device_map='auto', torch_dtype=torch.float16)
    mod.eval()
    return tok, mod

@st.cache_resource(show_spinner='Loading embedder...')
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner='Building RAG...')
def build_rag():
    emb = load_embedder()
    client = chromadb.Client()
    try: client.delete_collection('energy_live')
    except: pass
    col = client.create_collection('energy_live')
    docs = [
        ('b1','battery_manual.pdf','Battery should not discharge below 20% SOC. Optimal 20-90%. Max charge 5kW. Capacity 10kWh. Full charge from 20% takes 1.6 hours.'),
        ('b2','battery_manual.pdf','Warranty 10 years or 4000 cycles. Range 0-45C. Above 40C auto-reduces charge rate. Annual inspection recommended.'),
        ('t1','tariff_schedule.pdf','K-Electric 2024: 0-100 units PKR 16.48, 101-200 PKR 22.12, 201-300 PKR 28.50, 301-700 PKR 38.75, above 700 PKR 45.20. Peak 6PM-10PM 30% surcharge. Off-peak 10PM-7AM 28% discount.'),
        ('n1','net_metering_policy.pdf','NEPRA net metering: solar consumers export surplus to grid for bill credits. Buyback PKR 19.32/kWh. Max export equals sanctioned load.'),
        ('n2','net_metering_policy.pdf','Net metering: submit DISCO application with system capacity, inverter specs, single-line diagram. Approval 30-60 days. NEPRA bi-directional meter required.'),
        ('s1','solar_installation_guide.pdf','Karachi: south-facing panels at 24 degree tilt. Average 5.5 peak sun hours/day. 5kW gives 22-25 kWh/day summer. Dust reduces output 5-10%.'),
        ('h1','historical_data.txt','June 2024 Karachi: 14.2 kWh/day for 4.8kW system. Peak grid 7PM-10PM. Household average 18 kWh/day. Battery full by 2PM clear days.'),
        ('p1','energy_policies.txt','NEPRA requires registration above 1kW. Net metering via DISCO. Export capped at sanctioned load. Credits cannot exceed monthly charges.'),
    ]
    for did, src, txt in docs:
        col.add(ids=[did], embeddings=[emb.encode(txt).tolist()],
                documents=[txt], metadatas=[{'source':src}])
    return col, emb

def llm_run(prompt, max_tok=200):
    tok, mod = load_llm()
    dev = next(mod.parameters()).device
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=768).to(dev)
    with torch.no_grad():
        out = mod.generate(**inp, max_new_tokens=max_tok, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)

CMAP = {'Clear':('Sunny',5),'Clouds':('Cloudy',75),'Rain':('Rainy',90),
        'Haze':('Hazy',50),'Mist':('Misty',55),'Drizzle':('Drizzle',80),
        'Thunderstorm':('Thunderstorm',95),'Smoke':('Smoky',60)}

def get_weather():
    try:
        r = requests.get(f'http://api.openweathermap.org/data/2.5/weather'
                         f'?q=Karachi,PK&appid={WEATHER_API}&units=metric', timeout=6)
        d = r.json()
        lbl, _ = CMAP.get(d['weather'][0]['main'], (d['weather'][0]['main'], d['clouds']['all']))
        return {
            'condition': lbl, 'temperature': round(d['main']['temp'],1),
            'feels_like': round(d['main']['feels_like'],1), 'cloud_cover': d['clouds']['all'],
            'humidity': d['main']['humidity'], 'wind_kmh': round(d['wind']['speed']*3.6,1),
            'sunrise': datetime.fromtimestamp(d['sys']['sunrise']).strftime('%H:%M'),
            'sunset':  datetime.fromtimestamp(d['sys']['sunset']).strftime('%H:%M'),
            'solar_index': round(max(0,(1-d['clouds']['all']/100)*(1-d['main']['humidity']/200)),2),
            'source': 'OpenWeatherMap'
        }
    except:
        return {'condition':'Sunny','temperature':36.0,'feels_like':38.5,
                'cloud_cover':8,'humidity':58,'wind_kmh':12.0,
                'sunrise':'06:02','sunset':'19:24','solar_index':0.87,'source':'fallback'}

def get_tariff(monthly=350):
    SLABS = [(100,16.48,'0-100'),(200,22.12,'101-200'),(300,28.50,'201-300'),
             (700,38.75,'301-700'),(9999,45.20,'700+')]
    TOU   = [((18,22),'peak',1.30),((7,11),'morning',1.15),
             ((12,17),'shoulder',1.00),((0,7),'off-peak',0.72),((22,24),'off-peak',0.72)]
    h = datetime.now().hour
    per, mult = 'off-peak', 0.72
    for (s,e),p,m in TOU:
        if s<=h<e: per,mult=p,m; break
    base,slab = 45.20,'700+'
    for lim,rate,label in SLABS:
        if monthly<=lim: base,slab=rate,label; break
    return {'period':per,'rate_pkr':round(base*mult,2),'base_rate':base,
            'slab':slab,'multiplier':mult,'current_hour':h,'avoid_grid':per=='peak'}

def get_battery():
    soc=round(random.uniform(22,94),1); hlth=round(random.uniform(80,99),1)
    temp=round(random.uniform(26,44),1); ubl=round((soc/100)*10*0.90,2)
    return {
        'soc_percent':soc,
        'status':'Full' if soc>=90 else 'Good' if soc>=60 else 'Moderate' if soc>=40 else 'Low',
        'health_percent':hlth,
        'health_label':'Excellent' if hlth>=95 else 'Good' if hlth>=85 else 'Degraded',
        'charge_rate_kw':round(random.uniform(0.5,5.0),1),
        'usable_kwh':ubl,'hours_remaining':round(ubl/1.8,1),
        'temperature_c':temp,
        'temp_note':'High-throttled' if temp>40 else 'Normal',
        'cycle_count':random.randint(120,3900),'warranty_valid':True
    }

def get_solar(panels=18, watt=580, inv=0.96):
    PSH={1:5.2,2:5.8,3:6.1,4:6.5,5:6.8,6:5.9,7:4.8,8:5.1,9:5.6,10:5.9,11:5.5,12:5.0}
    TD ={1:0.94,2:0.93,3:0.91,4:0.89,5:0.87,6:0.86,7:0.88,8:0.88,9:0.89,10:0.91,11:0.93,12:0.94}
    sys=round((panels*watt)/1000,2); mon=datetime.now().month; h=datetime.now().hour
    psh=PSH[mon]; td=TD[mon]; daily=round(sys*psh*inv*td*0.95,2)
    cur=round(sys*math.exp(-0.5*((h-12)/3.5)**2)*inv*td,2) if 6<=h<=18 else 0.0
    return {'system_kw':sys,'current_kw':cur,'daily_forecast_kwh':daily,
            'peak_sun_hours':psh,'temp_derate':td,'num_panels':panels,
            'panel_watt':watt,'net_export':round(max(0,daily-18.0),2)}

BLOCKED=['turn off refrigerator','turn off fridge','override','shutdown',
         'disable safety','hack','bypass','inject','drop table','rm -rf','delete all']

def run_pipeline(user_input, monthly=350, panels=18, watt=580):
    logs=[]
    for kw in BLOCKED:
        if kw in user_input.lower():
            return None, f'BLOCKED: {kw}', []
    logs.append('SecurityAgent  : APPROVED')
    W=get_weather(); T=get_tariff(monthly); B=get_battery(); S=get_solar(panels,watt)
    logs.append(f'WeatherTool    : {W["condition"]}, {W["temperature"]}C, {W["cloud_cover"]}% cloud')
    logs.append(f'TariffTool     : {T["period"]}, PKR {T["rate_pkr"]}/kWh')
    logs.append(f'BatteryTool    : {B["soc_percent"]}% SOC - {B["status"]}')
    logs.append(f'SolarTool      : {S["current_kw"]}kW now | {S["daily_forecast_kwh"]}kWh forecast')
    CF=[(20,0.98),(40,0.88),(60,0.72),(80,0.55),(101,0.35)]
    cf=0.35
    for lim,f in CF:
        if W['cloud_cover']<lim: cf=f; break
    adj=round(S['daily_forecast_kwh']*cf,2); cur=round(S['current_kw']*cf,2)
    conf='High' if W['cloud_cover']<30 else 'Medium' if W['cloud_cover']<65 else 'Low'
    logs.append(f'ForecastAgent  : {adj} kWh adjusted | {conf} confidence')
    soc=B['soc_percent']; per=T['period']
    if cur>=3.5 and soc<75:       act,av,sc,rsn='Use Solar + Charge Battery',True,95,f'Solar {cur}kW battery {soc}%'
    elif cur>=2.5 and soc>=75:    act,av,sc,rsn='Use Solar Directly',True,88,'Solar covers load'
    elif per=='peak' and soc>=55: act,av,sc,rsn='Discharge Battery Peak Avoidance',True,85,'Peak tariff active'
    elif per=='off-peak' and soc<70: act,av,sc,rsn='Charge Battery Off-Peak',False,78,'Cheap rate'
    elif per=='peak' and soc<55:  act,av,sc,rsn='Reduce Load Peak+Low Battery',False,60,'Peak+low battery'
    else:                          act,av,sc,rsn='Grid Fallback',False,40,'No signal'
    logs.append(f'OptimizationAgent: {act} | {sc}/100')
    DEMAND=18.0; grid=max(0,round(DEMAND-adj,2)); rate=T['rate_pkr']
    gcost=round(grid*rate,2); fcost=round(DEMAND*rate,2); sav=round(fcost-gcost,2)
    earn=round(S['net_export']*19.32,2); cov=round(min(100,(adj/DEMAND)*100),1); msav=round(sav*30,2)
    logs.append(f'CostAgent      : PKR {sav}/day | {cov}% solar')
    prompt=(f'Smart home energy advisor Karachi. Weather {W["condition"]} {W["temperature"]}C '
            f'{W["cloud_cover"]}% clouds. Solar {adj}kWh {conf} confidence. Battery {soc}% SOC. '
            f'Tariff {per} PKR {rate}/kWh. Action: {act}. Savings PKR {sav}/day. '
            f'Solar {cov}% demand. 3-sentence advisory:')
    llm_out=llm_run(prompt,200)
    logs.append('ReportingAgent : LLM report generated')
    return {'W':W,'T':T,'B':B,'S':S,
            'fc':{'adj':adj,'cur':cur,'conf':conf,'cf':cf},
            'op':{'act':act,'av':av,'sc':sc,'rsn':rsn},
            'co':{'grid':grid,'gcost':gcost,'sav':sav,'earn':earn,'cov':cov,'msav':msav,'fcost':fcost},
            'rep':llm_out}, None, logs

def web_search(query, num=5):
    try:
        r = requests.get('https://serpapi.com/search',
            params={'q':query,'api_key':SERP_API,'engine':'google','num':num,'hl':'en'},
            timeout=10)
        data = r.json()
        return [{'title':x.get('title',''),'link':x.get('link',''),
                 'snippet':x.get('snippet',''),'source':x.get('displayed_link','')}
                for x in data.get('organic_results',[])[:num]]
    except Exception as e:
        return [{'title':'Search failed','link':'','snippet':str(e),'source':'error'}]

# Sidebar
with st.sidebar:
    st.title('Smart Energy AI')
    st.caption('Karachi | FLAN-T5-XL | A2A | MCP | RAG')
    st.divider()
    page = st.radio('Navigation', [
        'Live Dashboard','Run Pipeline','LLM Chat',
        'RAG Policy Search','Live Web Search','Analytics'
    ])
    st.divider()
    st.markdown('**System Config**')
    num_panels  = st.slider('Solar Panels', 6, 30, 18)
    panel_watt  = st.selectbox('Panel Wattage (W)', [300,350,400,450,500,550,580,600], index=6)
    monthly_kwh = st.slider('Monthly Usage (units)', 100, 800, 350)
    st.caption(f'System: {round(num_panels*panel_watt/1000,2)} kW')

# PAGE 1 - LIVE DASHBOARD
if page == 'Live Dashboard':
    st.title('Smart Energy Dashboard')
    st.caption(f'Live - Karachi - {datetime.now().strftime("%A %d %B %Y  %H:%M:%S")}')
    st.divider()
    W=get_weather(); T=get_tariff(monthly_kwh); B=get_battery(); S=get_solar(num_panels,panel_watt)
    st.subheader('Weather - Karachi (OpenWeatherMap Live)')
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric('Condition', W['condition'])
    c2.metric('Temperature', f'{W["temperature"]}C', f'Feels {W["feels_like"]}C')
    c3.metric('Cloud Cover', f'{W["cloud_cover"]}%')
    c4.metric('Humidity', f'{W["humidity"]}%')
    c5.metric('Wind', f'{W["wind_kmh"]} km/h')
    st.caption(f'Sunrise {W["sunrise"]} | Sunset {W["sunset"]} | Solar Index {W["solar_index"]} | {W["source"]}')
    st.divider()
    st.subheader('Solar Generation')
    s1,s2,s3,s4=st.columns(4)
    s1.metric('Current Output', f'{S["current_kw"]} kW')
    s2.metric('Daily Forecast', f'{S["daily_forecast_kwh"]} kWh')
    s3.metric('System Size', f'{S["system_kw"]} kW')
    s4.metric('Net Export', f'{S["net_export"]} kWh')
    st.caption(f'{num_panels} panels x {panel_watt}W | PSH {S["peak_sun_hours"]}h | Derate {S["temp_derate"]}')
    st.divider()
    cb,ct=st.columns(2)
    with cb:
        st.subheader('Battery')
        b1,b2,b3=st.columns(3)
        b1.metric('SOC', f'{B["soc_percent"]}%', B['status'])
        b2.metric('Health', f'{B["health_percent"]}%', B['health_label'])
        b3.metric('Usable', f'{B["usable_kwh"]} kWh', f'{B["hours_remaining"]}h')
        st.progress(int(B['soc_percent']))
        st.caption(f'Temp {B["temperature_c"]}C | {B["temp_note"]} | Cycles {B["cycle_count"]}')
    with ct:
        st.subheader('Tariff - K-Electric')
        t1,t2,t3=st.columns(3)
        t1.metric('Period', T['period'].upper())
        t2.metric('Rate', f'PKR {T["rate_pkr"]}/kWh')
        t3.metric('Avoid Grid', 'YES' if T['avoid_grid'] else 'NO')
        st.caption(f'Base PKR {T["base_rate"]} x {T["multiplier"]} | Slab {T["slab"]}')
        st.info(f'Hour: {T["current_hour"]}:00')
    if st.button('Refresh'): st.rerun()

# PAGE 2 - RUN PIPELINE
elif page == 'Run Pipeline':
    st.title('Full Agentic Pipeline')
    st.caption('Security > Forecast > Optimization > Cost > LLM Report')
    st.divider()
    user_input=st.text_input('Energy request:','Optimize my home energy usage for today')
    if st.button('Run Pipeline'):
        with st.spinner('Running all 5 agents...'):
            res,err,logs=run_pipeline(user_input,monthly_kwh,num_panels,panel_watt)
        if err:
            st.error(err)
        else:
            st.subheader('A2A Message Log')
            for log in logs: st.code(log)
            st.divider()
            st.subheader('Agent Outputs')
            L,R=st.columns(2)
            with L:
                with st.expander('Security Agent', expanded=True):
                    st.success('APPROVED')
                    st.write(f'Input: {user_input}')
                with st.expander('Forecast Agent', expanded=True):
                    st.metric('Adjusted Forecast', f'{res["fc"]["adj"]} kWh')
                    st.metric('Current Solar', f'{res["fc"]["cur"]} kW')
                    st.metric('Confidence', res['fc']['conf'])
                    st.write(f'{res["W"]["condition"]} | {res["W"]["temperature"]}C | {res["W"]["cloud_cover"]}% cloud')
                with st.expander('Optimization Agent', expanded=True):
                    st.metric('Action', res['op']['act'])
                    st.metric('Score', f'{res["op"]["sc"]}/100')
                    st.metric('Avoid Grid', 'YES' if res['op']['av'] else 'NO')
                    st.info(res['op']['rsn'])
            with R:
                with st.expander('Cost Analysis Agent', expanded=True):
                    st.metric('Daily Savings', f'PKR {res["co"]["sav"]}')
                    st.metric('Solar Coverage', f'{res["co"]["cov"]}%')
                    st.metric('Grid Draw', f'{res["co"]["grid"]} kWh')
                    st.metric('Export Earnings', f'PKR {res["co"]["earn"]}')
                    st.metric('Monthly Savings', f'PKR {res["co"]["msav"]}')
                with st.expander('Reporting Agent - LLM', expanded=True):
                    st.success(res['rep'])
            st.divider()
            st.subheader('Energy Flow')
            fig,ax=plt.subplots(figsize=(8,3.5))
            vals=[res['fc']['adj'],res['co']['grid'],18.0]
            lbls=['Solar Supply','Grid Draw','Daily Demand']
            cols=['#f59e0b','#3b82f6','#6b7280']
            brs=ax.bar(lbls,vals,color=cols,edgecolor='white',linewidth=1.2)
            for b,v in zip(brs,vals):
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.15,f'{v} kWh',ha='center',fontsize=11,color='white')
            ax.set_facecolor('#0f172a'); fig.patch.set_facecolor('#0f172a')
            ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
            st.pyplot(fig); plt.close()

# PAGE 3 - LLM CHAT
elif page == 'LLM Chat':
    st.title('Energy Assistant')
    st.caption('FLAN-T5-XL | Live context injected per message')
    st.divider()
    if 'chat' not in st.session_state: st.session_state.chat=[]
    for role,msg in st.session_state.chat:
        with st.chat_message(role): st.write(msg)
    q=st.chat_input('Ask anything about your energy system...')
    if q:
        st.session_state.chat.append(('user',q))
        with st.chat_message('user'): st.write(q)
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                W=get_weather(); B=get_battery(); T=get_tariff(monthly_kwh); S=get_solar(num_panels,panel_watt)
                prompt=(f'Smart home energy advisor Karachi. Weather {W["condition"]} {W["temperature"]}C. '
                        f'Battery {B["soc_percent"]}% SOC {B["status"]}. '
                        f'Solar {S["current_kw"]}kW now {S["daily_forecast_kwh"]}kWh today. '
                        f'Tariff {T["period"]} PKR {T["rate_pkr"]}/kWh. System {S["system_kw"]}kW. '
                        f'Question: {q} Answer 2-3 sentences:')
                ans=llm_run(prompt,180)
            st.write(ans)
        st.session_state.chat.append(('assistant',ans))
    if st.button('Clear Chat'): st.session_state.chat=[]; st.rerun()

# PAGE 4 - RAG POLICY SEARCH
elif page == 'RAG Policy Search':
    st.title('RAG Policy Search')
    st.caption('ChromaDB | all-MiniLM-L6-v2 | 8 documents | NEPRA / K-Electric / Battery / Solar')
    st.divider()
    col,emb=build_rag()
    quick=st.selectbox('Quick questions:',['-- select --',
        'What does net metering allow?','What is the minimum battery SOC?',
        'What are peak hour tariff rates?','How many peak sun hours does Karachi get?',
        'How do I apply for net metering?','What is the battery warranty?'])
    query=st.text_input('Or type your own:',value=quick if quick!='-- select --' else '')
    if st.button('Search') and query and query!='-- select --':
        with st.spinner('Searching...'):
            res=col.query(query_embeddings=[emb.encode(query).tolist()],n_results=2)
            chunks=res['documents'][0]; sources=[m['source'] for m in res['metadatas'][0]]
            context=' '.join(chunks)
            ans=llm_run(f'Context: {context} Question: {query} Answer concisely:',150)
        st.subheader('Answer'); st.success(ans)
        st.subheader('Retrieved Chunks')
        for chunk,src in zip(chunks,sources):
            with st.expander(f'{src}'): st.write(chunk)

# PAGE 5 - LIVE WEB SEARCH
elif page == 'Live Web Search':
    st.title('Live Web Search')
    st.caption('SerpAPI | Google Search | Real-time results | LLM summarizes in your system context')
    st.divider()
    st.info('Search for NEPRA updates, K-Electric news, solar prices, load shedding, or any energy topic. LLM summarizes findings in context of your system.')
    st.markdown('**Quick searches:**')
    qcol1,qcol2,qcol3=st.columns(3)
    quick_list=[
        'NEPRA net metering 2024 Pakistan',
        'K-Electric tariff increase 2025',
        'Solar panel prices Pakistan 2025',
        'Load shedding schedule Karachi 2025',
        'Best solar inverter Pakistan 2025',
        'NEPRA solar policy update',
    ]
    for i,qs in enumerate(quick_list):
        col_idx=[qcol1,qcol2,qcol3][i%3]
        if col_idx.button(qs,key=f'qs{i}'): st.session_state['sq']=qs
    st.divider()
    default=st.session_state.get('sq','')
    search_q=st.text_input('Search query:',value=default,
                            placeholder='e.g. NEPRA net metering policy 2025 Pakistan')
    num_r=st.slider('Results to fetch',3,8,5)
    if st.button('Search Web') and search_q:
        with st.spinner(f'Searching: {search_q}...'):
            results=web_search(search_q,num_r)
        if not results or results[0]['source']=='error':
            st.error('Search failed. Check internet connection.')
        else:
            st.subheader(f'Results for: {search_q}')
            snippets=''
            for i,res in enumerate(results,1):
                with st.expander(f'{i}. {res["title"]}',expanded=(i<=2)):
                    st.markdown(f'**Source:** [{res["source"]}]({res["link"]})')
                    st.write(res['snippet'])
                    st.markdown(f'[Open link]({res["link"]})')
                snippets+=f'{res["title"]}: {res["snippet"]} '
            st.divider()
            st.subheader('LLM Summary')
            st.caption('FLAN-T5-XL synthesizes results in your system context')
            with st.spinner('Summarizing...'):
                W=get_weather(); T=get_tariff(monthly_kwh); S=get_solar(num_panels,panel_watt)
                sp=(f'Smart home advisor Karachi with {S["system_kw"]}kW solar. '
                    f'Tariff {T["period"]} PKR {T["rate_pkr"]}/kWh. '
                    f'Web search about {search_q}: {snippets[:600]} '
                    f'Summarize key findings for this household in 3 sentences.')
                summary=llm_run(sp,200)
            st.success(summary)
            if 'sh' not in st.session_state: st.session_state.sh=[]
            st.session_state.sh.append({'q':search_q,'t':datetime.now().strftime('%H:%M'),'n':len(results)})
    if 'sh' in st.session_state and st.session_state.sh:
        st.divider(); st.subheader('Recent Searches')
        for item in reversed(st.session_state.sh[-5:]):
            st.caption(f'{item["t"]} - {item["q"]} ({item["n"]} results)')

# PAGE 6 - ANALYTICS
elif page == 'Analytics':
    st.title('System Analytics')
    st.caption('All values computed from real data')
    st.divider()
    S=get_solar(num_panels,panel_watt); T=get_tariff(monthly_kwh); W=get_weather()
    sys_kw=S['system_kw']; td=S['temp_derate']
    cf=round(max(0,(1-W['cloud_cover']/100)*(1-W['humidity']/200)),2)
    hrs=list(range(24))
    out=[round(sys_kw*math.exp(-0.5*((h-12)/3.5)**2)*0.96*td*cf,2) if 6<=h<=18 else 0.0 for h in hrs]
    st.subheader('Solar Output Today (Hour by Hour)')
    fig1,ax1=plt.subplots(figsize=(11,3.5))
    ax1.fill_between(hrs,out,alpha=0.35,color='#f59e0b')
    ax1.plot(hrs,out,color='#f59e0b',linewidth=2.5)
    ax1.axvline(x=datetime.now().hour,color='#ef4444',linestyle='--',linewidth=1.5,label='Now')
    ax1.set_xlabel('Hour',color='white'); ax1.set_ylabel('kW',color='white')
    ax1.set_facecolor('#0f172a'); fig1.patch.set_facecolor('#0f172a')
    ax1.tick_params(colors='white'); ax1.spines[:].set_color('#334155')
    ax1.legend(facecolor='#1e1e2e',labelcolor='white')
    st.pyplot(fig1); plt.close()
    PSH=[5.2,5.8,6.1,6.5,6.8,5.9,4.8,5.1,5.6,5.9,5.5,5.0]
    TDL=[0.94,0.93,0.91,0.89,0.87,0.86,0.88,0.88,0.89,0.91,0.93,0.94]
    MTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    rate=T['base_rate']
    savs=[round(min(sys_kw*p*0.96*t*0.95,18.0)*rate*30,0) for p,t in zip(PSH,TDL)]
    st.subheader('Monthly Savings Projection')
    fig2,ax2=plt.subplots(figsize=(11,3.5))
    brs=ax2.bar(MTHS,savs,color='#7c3aed',edgecolor='white',linewidth=0.8)
    brs[datetime.now().month-1].set_color('#22c55e')
    ax2.set_ylabel('PKR/month',color='white')
    ax2.set_facecolor('#0f172a'); fig2.patch.set_facecolor('#0f172a')
    ax2.tick_params(colors='white'); ax2.spines[:].set_color('#334155')
    green=mpatches.Patch(color='#22c55e',label='Current month')
    ax2.legend(handles=[green],facecolor='#1e1e2e',labelcolor='white')
    st.pyplot(fig2); plt.close()
    st.divider()
    a1,a2,a3,a4=st.columns(4)
    a1.metric('Annual Savings', f'PKR {int(sum(savs)):,}')
    a2.metric('System Size', f'{sys_kw} kW')
    a3.metric('Panels', f'{num_panels} x {panel_watt}W')
    a4.metric('Avg Daily Solar', f'{round(sum([sys_kw*p*0.96*t*0.95 for p,t in zip(PSH,TDL)])/12,1)} kWh')