from DDChatbot.intentions import intentions_funcs

intentions_dict = {
    1: {"intention_name": "I want to create or add more data to an incident report", "function": intentions_funcs.create_ir},
    2: {"intention_name": "I want to print the incident report", "function": intentions_funcs.print_ir},
    3: {"intention_name": "I want to share the incident report with my managers", "function": intentions_funcs.share_ir},
    4: {"intention_name": "I want to see all incident information", "function": intentions_funcs.see_all_data}
}