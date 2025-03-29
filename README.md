
# Movie Recommendation System

## Overview

This project implements a movie recommendation system using collaborative filtering and enhanced similarity measures. The system evaluates the performance of different recommendation approaches based on metrics like precision, recall, F1-score, MAE, and RMSE. Additionally, it introduces enhancements to improve recommendation quality by combining user similarity and demographic data.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Data](#data)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Setup and Usage](#setup-and-usage)
6. [Metrics and Evaluation](#metrics-and-evaluation)
7. [Results](#results)
8. [Contributing](#contributing)

---

## Project Structure
- **main.py**: Contains the core implementation of the recommendation system.
- **README.md**: Provides an overview of the project (this file).
- **Data Files**:
  - `u.data`: Ratings dataset.
  - `u.item`: Movie details.
  - `u.user`: User details.
  - `u1.base` and `u1.test`: Base and test datasets for training and evaluation.

---

## Data
The dataset used in this project comes from the [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/). It includes:
- **Ratings**: User ratings for movies.
- **Movies**: Metadata for movies including genres and titles.
- **Users**: Demographic information such as age, gender, and occupation.

---

## Features
### Core Features:
1. **Recommendation Approaches**:
   - **Collaborative Filtering**: Using cosine similarity to find similar users.
   - **Random Recommendations**: Baseline for comparison.
2. **Evaluation Metrics**:
   - Precision, Recall, F1-score.
   - Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
3. **Top-N Recommendations**:
   - Generates top-rated movies based on weighted ratings.

### Advanced Features:
1. **Enhanced Similarity**:
   - Combines demographic data (age) with cosine similarity for improved recommendations.
2. **Dynamic Weighting**:
   - Evaluates different weight combinations for similarity metrics.

---

## Requirements
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `sklearn`

Install the required packages using:
```bash
pip install -r requirements.txt
```

---

## Setup and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Place Dataset**:
   Place the MovieLens dataset files (`u.data`, `u.item`, `u.user`, etc.) in the project directory.

3. **Run the Code**:
   Execute the script:
   ```bash
   python main.py
   ```

4. **Results**:
   The system will output:
   - Top-rated movies.
   - Precision, Recall, and F1-scores for different methods.
   - Enhanced recommendation results.

---

## Metrics and Evaluation

### Evaluation Metrics:
- **Precision**: Ratio of relevant recommendations to total recommendations.
- **Recall**: Ratio of relevant recommendations to total relevant items.
- **F1-Score**: Harmonic mean of precision and recall.
- **MAE (Mean Absolute Error)**: Measures the average magnitude of prediction errors.
- **RMSE (Root Mean Square Error)**: Measures the standard deviation of prediction errors.

### Process:
1. Hide 20% of ratings for testing.
2. Predict ratings for hidden cells.
3. Compare predicted ratings with actual ratings.

---

## Results
### Question 2 Results:
- **Top 10 Movies**: Based on Weighted Rating (WR).
- **Random Recommendations**: 10 randomly selected movies.
- **System vs Random Recommendations**:
  - F1-Score comparison for both approaches.

### Question 3 Results:
- User-based collaborative filtering using cosine similarity.
- Metrics: MAE, RMSE, Precision, Recall, F1.

### Question 4 Results:
- Enhanced similarity by combining cosine similarity with age similarity.
- Evaluation for different weight combinations:
  - Precision, Recall, and F1-score improvements.

##DEVELOPED BY

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request. Ensure your code is well-documented and follows the project structure.


