import pandas as pd

def generate_report(metadata_path):

    df = pd.read_csv(metadata_path)

    report = df.groupby("speaker").apply(
        lambda x: (x["end"] - x["start"]).sum()
    )

    report = report.reset_index()
    report.columns = ["speaker", "total_speech_seconds"]

    report.to_csv("output/speaker_report.csv", index=False)

    print(report)