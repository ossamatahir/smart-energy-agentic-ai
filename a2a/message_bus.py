
import json
from datetime import datetime

class MessageBus:
    def __init__(self):
        self.messages = []

    def send(self, from_agent, to_agent, msg_type, data):
        self.messages.append({
            "from"     : from_agent,
            "to"       : to_agent,
            "type"     : msg_type,
            "data"     : data,
            "timestamp": datetime.now().isoformat()
        })

    def get_messages(self, to_agent=None):
        if to_agent:
            return [m for m in self.messages if m["to"] == to_agent]
        return self.messages

    def clear(self):
        self.messages = []
