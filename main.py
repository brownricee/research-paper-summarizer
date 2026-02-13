def main():
    pdf_path = parse_args()

    print("Starting extraction...")
    text = extract_text(pdf_path)

    print("Pipeline complete.")
