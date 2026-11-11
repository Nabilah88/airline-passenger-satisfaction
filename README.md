# Airline Passenger Satisfaction Prediction

Machine learning project for the [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) midterm (DataTalks.Club).

## Problem description

Airlines collect survey feedback to understand how passengers experienced a flight, but surveys only reach some passengers and arrive after the fact. This project builds a binary classifier that predicts whether a passenger will be **satisfied** or **neutral/dissatisfied**, based on their profile, trip details, flight delays and ratings of services such as online boarding, wifi and seat comfort.

How it could be used: an airline could use the model to estimate satisfaction for passengers who did not answer a survey, and to see which services are most associated with satisfaction, which helps decide where to improve. The model is a learning project and a decision-support tool, not a replacement for real feedback.

## Dataset

- Source: <Kaggle dataset name and link>
- Files: `data/train.csv` (103,904 rows) and `data/test.csv`, both committed in this repo.
- Target: `satisfaction` (satisfied / neutral or dissatisfied).
- Features (22): gender, customer type, age, type of travel, class, flight distance, 14 service ratings (1-5, where 0 means "not applicable"), and departure and arrival delay in minutes.

## Approach

1. **Data cleaning:** dropped the index and id columns, standardized column names and text values, and filled 310 missing arrival delays with the departure delay (the two correlate at 0.96).
2. **EDA:** target balance, distributions, satisfaction by customer type, travel type and class, correlations with the target, and multicollinearity (only two feature pairs above 0.7).
3. **Models:** logistic regression, random forest and XGBoost, each tuned on a validation set.
4. **Final model:** XGBoost (`max_depth=7`, `learning_rate=0.3`, 100 trees), trained on all of `train.csv`.

### Results

| Model                       | Validation AUC |
| --------------------------- | -------------- |
| Logistic regression (tuned) | 0.927          |
| Random forest (tuned)       | 0.994          |
| XGBoost (tuned)             | 0.995          |

On the held-out `test.csv`, the final model scores **AUC 0.995** and **accuracy 0.963**. These match the validation results. I checked for leakage: the target is not among the features, and no rows are shared between the train and test files.

The most important features for the model are online boarding, type of travel, inflight wifi, customer type and class. These show what the model relies on, not what causes satisfaction.

## Repository structure

| File               | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| `notebook.ipynb`   | Data cleaning, EDA, model selection and tuning     |
| `train.py`         | Trains the final model and saves it to `model.bin` |
| `predict.py`       | Flask web service that serves predictions          |
| `predict_test.py`  | Sends an example request to the service            |
| `model.bin`        | Saved DictVectorizer and XGBoost model             |
| `Dockerfile`       | Container for the web service                      |
| `requirements.txt` | Python dependencies                                |
| `data/`            | Training and test data                             |

## How to run

### 1. Set up the environment

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/Scripts/activate    # Windows (Git Bash); on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 was used. To run the notebook, also install `jupyter`, `matplotlib` and `seaborn`.

### 2. Train the model

```bash
python train.py
```

This prints the test AUC (about 0.995) and saves `model.bin`.

### 3. Run the service locally

```bash
python predict.py
```

In a second terminal:

```bash
python predict_test.py
```

### 4. Run with Docker

```bash
docker build -t airline-satisfaction .
docker run -it --rm -p 9696:9696 airline-satisfaction
```

Then run `python predict_test.py` from another terminal.

## Example

Request (POST to `http://localhost:9696/predict`): a JSON object with the 22 features, for example:

```json
{
  "gender": "female",
  "customer_type": "loyal_customer",
  "age": 42,
  "type_of_travel": "personal_travel",
  "class": "eco",
  "flight_distance": 800,
  "inflight_wifi_service": 2,
  "departure_arrival_time_convenient": 3,
  "ease_of_online_booking": 3,
  "gate_location": 3,
  "food_and_drink": 3,
  "online_boarding": 3,
  "seat_comfort": 3,
  "inflight_entertainment": 3,
  "on_board_service": 3,
  "leg_room_service": 3,
  "baggage_handling": 3,
  "checkin_service": 3,
  "inflight_service": 3,
  "cleanliness": 3,
  "departure_delay_in_minutes": 15,
  "arrival_delay_in_minutes": 20
}
```

Response:

```json
{ "satisfaction_probability": 0.0027, "satisfied": false }
```

Text values must be lowercase with underscores (for example `business_travel`), and all 22 fields must be present, because a missing or misspelled field does not raise an error.

## Cloud deployment

<URL, or a screenshot or video of the deployed service>
