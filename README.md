# 🌱 Predicting Seasons from Power Consumption in Denmark

This project explores how deep learning models can classify seasons of the year (winter, spring, summer, autumn) based on daily power consumption and renewable energy production in Denmark.

## 📊 Dataset

We use a subset of the [Open Power System Data](https://data.open-power-system-data.org/time_series/2020-10-06) dataset, focusing on three features:

- `DK_load_actual_entsoe_transparency` (Total power load)
- `DK_wind_generation_actual`
- `DK_solar_generation_actual`

The data is aggregated to daily records with 24 hourly values per feature.

## 🧠 Models

We implemented and evaluated three models using PyTorch:

1. **MLP** — Simple dense network baseline (Accuracy: 0.806)
2. **1D CNN** — Temporal pattern extraction from time series (Accuracy: 0.857)
3. **2D CNN** — Using GAF transformation to convert time series into images (Accuracy: 0.835)

## 📂 Folder Structure

- `notebooks/main.ipynb` — Main notebook with all experiments
- `data/` — Contains the zipped dataset
- `reports/` — Final report and assignment instructions
- `requirements.txt` — Python dependencies

## 🚀 How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/predict-season-power-denmark.git
cd predict-season-power-denmark

# Step 2: Set up the environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 3: Open and run the notebook
jupyter notebook notebooks/main.ipynb
```

## 📈 Results

The 1D CNN outperformed other models due to its ability to capture temporal dependencies.

## 📝 Future Work

- Incorporate external features like weather
- Experiment with hybrid CNN + LSTM models
- Try larger datasets for better generalization

## 🧑‍💻 Author

Ilyas Galiev — [il.galiev@innopolis.university](mailto:il.galiev@innopolis.university)
