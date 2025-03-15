def isValidJSONFile(file):
    if not file.filename.endswith(".json") or file.content_type != "application/json":
        return False
    return True
    
def isAllColumnsPresent(tickets):
    REQUIRED_FIELDS = {"description", "product"}
    if any(REQUIRED_FIELDS - ticket.keys() for ticket in tickets):
        return False
    return True

def isEmptyFile(tickets):
    if not tickets:
        return True
    return False