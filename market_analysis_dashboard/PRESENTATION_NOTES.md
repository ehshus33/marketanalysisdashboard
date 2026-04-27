# Presentation help

## What to say about the app
This application analyses one stock at a time using historical market data from Yahoo Finance. It calculates technical indicators, trains machine learning models, and predicts the next trading day's likely direction and return.

## What makes it stronger than the simple version
- Better visual design
- More charts
- Clearer interpretation
- Proper evaluation tab
- Advanced neural network option

## How to defend the model choices
I used simpler models such as Logistic Regression and Linear Regression because they are easier to interpret and justify. I also added a neural network as an advanced comparison model.

## How to defend accuracy
Accuracy is measured using a chronological train/test split, so the model is trained on older data and evaluated on newer unseen data. This avoids leakage and is more appropriate for financial time series.
