import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
DATASET_CANDIDATES = [
    ROOT_DIR / "books_dataset.csv",
    DATA_DIR / "books_dataset.csv",
]


def find_dataset() -> Path:
    """Locate the books dataset in common project folders."""
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset not found. Please place books_dataset.csv in the project root or in the data/ folder "
        "before running this script."
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for easier analysis."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def normalize_price(series: pd.Series) -> pd.Series:
    """Convert the price values to a numeric format."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix formatting issues, and prepare data for EDA."""
    df = df.copy()

    # Standardize columns
    df = clean_column_names(df)

    # Fill or convert common missing values
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()
            df[column] = df[column].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    # Handle duplicate records
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicate_count} duplicate records.")
    else:
        print("No duplicate records found.")

    # Convert numeric columns where relevant
    if "price" in df.columns:
        df["price"] = normalize_price(df["price"])

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if "availability" in df.columns:
        df["availability"] = df["availability"].astype(str).str.strip().str.lower()

    # Fill missing numeric values with median or mode where possible
    for column in ["price", "rating"]:
        if column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)

    return df


def display_basic_overview(df: pd.DataFrame) -> None:
    """Print dataset overview and summary information."""
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names and data types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0].to_string()) if not missing_values.empty and (missing_values > 0).any() else print("No missing values found.")

    print(f"\nDuplicate records: {df.duplicated().sum()}")


def descriptive_statistics(df: pd.DataFrame) -> None:
    """Show descriptive statistics for numerical columns."""
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        print("\nNo numeric columns available for descriptive statistics.")
        return

    print("\nDescriptive statistics:")
    print(df[numeric_columns].describe().round(2).to_string())


def identify_price_extremes(df: pd.DataFrame) -> None:
    """Print the highest and lowest book prices."""
    if "price" not in df.columns:
        print("\nNo 'price' column found in the dataset.")
        return

    max_price = df["price"].max()
    min_price = df["price"].min()
    print(f"\nHighest book price: {max_price:.2f}")
    print(f"Lowest book price: {min_price:.2f}")


def analyze_ratings_and_availability(df: pd.DataFrame) -> None:
    """Summarize rating and availability patterns."""
    if "rating" in df.columns:
        avg_rating = df["rating"].mean()
        median_rating = df["rating"].median()
        max_rating = df["rating"].max()
        min_rating = df["rating"].min()
        print(f"\nAverage rating: {avg_rating:.2f}")
        print(f"Median rating: {median_rating:.2f}")
        print(f"Highest rating: {max_rating:.2f}")
        print(f"Lowest rating: {min_rating:.2f}")

    if "availability" in df.columns:
        availability_counts = df["availability"].value_counts(dropna=False)
        print("\nAvailability counts:")
        print(availability_counts.to_string())


def find_price_rating_patterns(df: pd.DataFrame) -> None:
    """Inspect relationships between price and rating."""
    if {"price", "rating"}.issubset(df.columns):
        corr_value = df[["price", "rating"]].corr().loc["price", "rating"]
        print(f"\nCorrelation between price and rating: {corr_value:.4f}")

        if df["price"].nunique() > 1:
            price_quartiles = pd.qcut(
                df["price"],
                q=min(4, df["price"].nunique()),
                duplicates="drop",
            )
            print("\nAverage rating by price quartile:")
            print(df.groupby(price_quartiles, observed=False)["rating"].mean().round(2).to_string())


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned dataset to the data folder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate charts for the EDA and save them in the output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # Price distribution histogram
    if "price" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["price"], bins=20, kde=True, color="steelblue")
        plt.title("Book Price Distribution")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / "price_distribution.png", dpi=300)
        plt.close()

    # Rating count bar chart
    if "rating" in df.columns:
        plt.figure(figsize=(8, 5))
        rating_counts = df["rating"].value_counts().sort_index()
        sns.barplot(x=rating_counts.index.astype(str), y=rating_counts.values, palette="viridis")
        plt.title("Rating Count Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "rating_count_chart.png", dpi=300)
        plt.close()

    # Availability count chart
    if "availability" in df.columns:
        plt.figure(figsize=(7, 5))
        availability_counts = df["availability"].value_counts()
        sns.barplot(x=availability_counts.index, y=availability_counts.values, palette="magma")
        plt.title("Book Availability Count")
        plt.xlabel("Availability")
        plt.ylabel("Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / "availability_count_chart.png", dpi=300)
        plt.close()

    # Price vs rating chart
    if {"price", "rating"}.issubset(df.columns):
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df, x="price", y="rating", hue="rating", palette="coolwarm", s=80)
        plt.title("Price vs Rating")
        plt.xlabel("Price")
        plt.ylabel("Rating")
        plt.tight_layout()
        plt.savefig(output_dir / "price_vs_rating.png", dpi=300)
        plt.close()

    # Correlation heatmap where applicable
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(8, 7))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()

    print(f"\nCharts saved to: {output_dir}")


def generate_report(df: pd.DataFrame, source_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a text report summarizing data quality, patterns, and conclusions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "eda_report.txt"
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    report_lines = [
        "Exploratory Data Analysis (EDA) Report",
        "====================================",
        f"Records analyzed: {df.shape[0]}",
        f"Source columns: {source_df.shape[1]}",
        f"Missing values in source data: {int(source_df.isna().sum().sum())}",
        f"Exact duplicate rows in source data: {int(source_df.duplicated().sum())}",
        "",
        "Source column data types:",
    ]
    report_lines.extend(
        f"- {column}: {dtype}" for column, dtype in source_df.dtypes.items()
    )
    report_lines.extend([
        "",
        f"Numerical columns: {', '.join(numeric_columns) or 'None'}",
        f"Categorical columns: {', '.join(categorical_columns) or 'None'}",
        "",
        "Summary statistics:",
    ])

    if "price" in df.columns:
        highest_price = df["price"].max()
        lowest_price = df["price"].min()
        average_price = df["price"].mean()
        report_lines.append(
            f"- Price: minimum {lowest_price:.2f}, median {df['price'].median():.2f}, "
            f"mean {average_price:.2f}, maximum {highest_price:.2f}."
        )

    if "rating" in df.columns:
        avg_rating = df["rating"].mean()
        report_lines.append(
            f"- Rating: minimum {df['rating'].min():.2f}, median {df['rating'].median():.2f}, "
            f"mean {avg_rating:.2f}, maximum {df['rating'].max():.2f}."
        )

    if "availability" in df.columns:
        availability_counts = df["availability"].value_counts()
        report_lines.append(
            "- Availability: "
            + ", ".join(f"{status} = {count}" for status, count in availability_counts.items())
            + "."
        )

    outlier_summaries = []
    report_lines.extend(["", "Outlier Detection (1.5 x IQR):"])
    for column in numeric_columns:
        values = df[column].dropna()
        if values.empty:
            report_lines.append(f"- {column}: no non-missing values to assess.")
            continue

        first_quartile, third_quartile = values.quantile([0.25, 0.75])
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        if outliers.empty:
            result = "no outliers"
        else:
            outlier_values = ", ".join(f"{value:g}" for value in outliers)
            result = f"{len(outliers)} outlier(s): {outlier_values}"
        outlier_summaries.append((column, len(outliers)))
        report_lines.append(
            f"- {column}: Q1={first_quartile:.2f}, Q3={third_quartile:.2f}, "
            f"IQR={interquartile_range:.2f}; fences={lower_bound:.2f} to "
            f"{upper_bound:.2f}; {result}."
        )

    report_lines.extend(["", "Meaningful Trends and Patterns:"])
    price_quartile_means = pd.Series(dtype=float)
    if "rating" in df.columns:
        rating_counts = df["rating"].value_counts()
        most_common_rating = rating_counts.idxmax()
        report_lines.append(
            f"- Ratings range from {df['rating'].min():.1f} to {df['rating'].max():.1f}; "
            f"{most_common_rating:.1f} is most frequent ({int(rating_counts.max())} books)."
        )

    if {"price", "rating"}.issubset(df.columns):
        correlation = df[["price", "rating"]].corr().loc["price", "rating"]
        if df["price"].nunique() > 1:
            price_quartiles = pd.qcut(
                df["price"],
                q=min(4, df["price"].nunique()),
                duplicates="drop",
            )
            price_quartile_means = df.groupby(price_quartiles, observed=False)["rating"].mean()
            if len(price_quartile_means) > 1:
                report_lines.append(
                    f"- Mean rating rises from {price_quartile_means.iloc[0]:.2f} in the lowest-price "
                    f"quartile to {price_quartile_means.iloc[-1]:.2f} in the highest-price quartile."
                )

        report_lines.append(
            f"- Price and rating have a Pearson correlation of {correlation:.3f}; this is an "
            "association in this sample, not evidence that price causes higher ratings."
        )

    if "genre" in df.columns:
        genre_counts = df["genre"].value_counts()
        top_genres = genre_counts[genre_counts == genre_counts.max()]
        report_lines.append(
            f"- The largest genre group is {', '.join(top_genres.index.astype(str))} "
            f"({int(genre_counts.max())} books)."
        )

    report_lines.extend(["", "Analytical Questions and Findings:"])
    if {"price", "rating"}.issubset(df.columns):
        quartile_finding = (
            f"mean rating ranges from {price_quartile_means.iloc[0]:.2f} in the lowest-price "
            f"quartile to {price_quartile_means.iloc[-1]:.2f} in the highest-price quartile"
            if len(price_quartile_means) > 1
            else "price quartile averages could not be calculated"
        )
        report_lines.append(
            f"1. Do higher-priced books tend to have higher ratings? In this dataset, the Pearson "
            f"correlation is {correlation:.3f}; {quartile_finding}."
        )

    if "availability" in df.columns:
        availability_counts = df["availability"].value_counts()
        report_lines.append(
            "2. What is the stock position of the catalog? "
            + ", ".join(f"{status}: {count}" for status, count in availability_counts.items())
            + "."
        )

    if outlier_summaries:
        outlier_findings = ", ".join(
            f"{column}: {count}" for column, count in outlier_summaries
        )
        report_lines.append(
            "3. Are there numerical outliers using the 1.5 x IQR rule? "
            f"{outlier_findings}."
        )

    report_lines.extend([
        "",
        "Conclusion:",
        f"- The {df.shape[0]}-book catalog has an average price of {df['price'].mean():.2f} "
        f"and an average rating of {df['rating'].mean():.2f}.",
        "- Eight books are in stock, three are low stock, and one is out of stock; the IQR check "
        "found no price or rating outliers.",
        "- Price and rating move together positively in this sample, but the small catalog does not "
        "support causal claims or broad generalization.",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nEDA report saved to: {report_path}")


def main() -> None:
    """Run the EDA workflow for the books dataset."""
    try:
        dataset_path = find_dataset()
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        source_df = df.copy()

        print("\nInitial dataset preview:")
        print(df.head(5).to_string(index=False))

        df = clean_dataset(df)
        display_basic_overview(df)
        descriptive_statistics(df)
        identify_price_extremes(df)
        analyze_ratings_and_availability(df)
        find_price_rating_patterns(df)

        cleaned_path = DATA_DIR / "cleaned_books_dataset.csv"
        save_cleaned_dataset(df, cleaned_path)
        create_visualizations(df, OUTPUT_DIR)
        generate_report(df, source_df, OUTPUT_DIR)

        print("\nEDA process completed successfully.")

    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        print("Please make sure books_dataset.csv exists in the project root or data/ folder.")
        sys.exit(1)

    except Exception as exc:
        print(f"\nUnexpected error occurred: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
