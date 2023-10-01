import json


def create_ir(self, memory):
    
    memory["new_report_data"] = self.update_report(memory["raw_question"], memory["starting_report_data"] , memory["schema"])
    return memory

def print_ir(self, memory):
    return "print_ir"

def share_ir(self, memory):
    return "share_ir"

def see_all_data(self, memory):
    return "see_all_data"
