import pandas as pd
import joblib
import sys
import time

def main(input_path, output_path):
    start = time.time()

    model = joblib.load("models/baseline.joblib")

    df = pd.read_csv(input_path)
    X = df[["x1", "x2"]]

    df["prediction"] = model.predict_proba(X)[:, 1]

    df.to_csv(output_path, index=False)

    print(f"Processed {len(df)} rows")
    print(f"Saved predictions to {output_path}")
    print(f"Took {time.time() - start:.3f} seconds")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
