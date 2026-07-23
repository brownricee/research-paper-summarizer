def aggregate_summary(section_summaries, section_order=None):
    if not section_summaries:
        return ""

    # If a paper-order list is provided, use it; otherwise fall back to dict order.
    if section_order is None:
        section_order = list(section_summaries.keys())

    output = ""
    for section_name in section_order:
        if section_name not in section_summaries:
            continue
        summary = section_summaries[section_name]
        output += summary
    return output
