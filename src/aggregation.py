def aggregate_summary(section_summaries, section_order=None, with_headers=False):
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
        # skips headers that have no content
        if not summary.strip():
            continue

        if with_headers:
            output += f"## {section_name.title()}\n{summary.strip()}\n\n"
        else:
            # Needs the trailing space since without it consecutive section summaries
            # weld together and the sentence boundary is lost for everything downstream.
            output += summary.strip() + " "

    return output.strip()
