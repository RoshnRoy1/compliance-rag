# eval/dataset.py

EVAL_SET = [
    {
        "question": "Within how many hours must a personal data breach be notified to the supervisory authority under GDPR?",
        "ground_truth": "Not later than 72 hours after having become aware of it, under Article 33 of the GDPR.",
        "expected_source": "gdpr.pdf",
    },
    {
        "question": "What is the maximum administrative fine for GDPR violations?",
        "ground_truth": "Up to 20,000,000 EUR, or 4% of total worldwide annual turnover of the preceding financial year, whichever is higher.",
        "expected_source": "gdpr.pdf",
    },
    {
        "question": "What counts as a high-risk AI system under the EU AI Act?",
        "ground_truth": "Stand-alone AI systems that pose a high risk of harm to health, safety, or fundamental rights based on their intended purpose, including AI systems used in law enforcement contexts.",
        "expected_source": "eu_ai_act.pdf",
    },
    {
        "question": "Does the EU AI Act apply to open source AI models?",
        "ground_truth": "Generally yes, though free and open-source models are exempt from certain obligations under Article 53, unless the model presents systemic risk.",
        "expected_source": "eu_ai_act.pdf",
    },
    {
        "question": "What rights do data subjects have under GDPR?",
        "ground_truth": "Rights include access, rectification, erasure, restriction of processing, data portability, and objection, covered under Articles 12 to 22.",
        "expected_source": "gdpr.pdf",
    },
    {
        "question": "What does Article 33 of the UK Data Protection Act reference?",
        "ground_truth": "Article 33 relates to notification of a personal data breach to the Commissioner, referenced within the UK GDPR provisions incorporated by the Data Protection Act 2018.",
        "expected_source": "uk_data_protection.pdf",
    },
    {
        "question": "What is a Data Protection Officer and when must one be appointed?",
        "ground_truth": "A Data Protection Officer must be appointed by controllers and processors under certain conditions specified in GDPR, including when core activities involve large-scale monitoring or processing of special category data.",
        "expected_source": "gdpr.pdf",
    },
    {
        "question": "How does the UK GDPR differ from the EU GDPR on maximum fines?",
        "ground_truth": "In relation to infringements before IP completion day, the UK substituted 17,500,000 pounds with 20 million euros for maximum fines.",
        "expected_source": "uk_data_protection.pdf",
    },
    {
        "question": "What is required for consent to be valid under GDPR?",
        "ground_truth": "Consent must be a clear affirmative act, freely given, specific, informed, and unambiguous, and the controller must be able to demonstrate it was given.",
        "expected_source": "gdpr.pdf",
    },
    {
        "question": "Compare data breach notification timelines between GDPR and the UK Data Protection Act.",
        "ground_truth": "Both reference the same 72-hour notification requirement to the supervisory authority (Commissioner in the UK), as the UK DPA incorporates the UK GDPR's Article 33 and 34 provisions.",
        "expected_source": "both",
    },
]