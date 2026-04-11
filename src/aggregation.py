def aggregate_summary(section_summaries):
    if not section_summaries:
        return ""

    output = ""
    for section_name, summary in section_summaries.items():
        output += f"## {section_name.title()}\n{summary}\n"
    return output
