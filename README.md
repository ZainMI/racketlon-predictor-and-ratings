# Racketlon Match Predictor & Rating System

**Author:** Zain Magdon-Ismail  

## Overview
This project aims to build a data-driven system that predicts the outcome of racketlon matches using historical performance data. To support accurate predictions, the project also includes the development of a player rating system that captures relative skill across disciplines and over time.

If completed within scope, the predictor will be deployed through a simple web interface that allows users to explore predicted match outcomes and player ratings.

---

## Tech Stack

### Backend / Data Science
- Python
  - Web scraping
  - Data cleaning and preprocessing
  - Feature engineering
  - Machine learning models

### Frontend (Optional)
- Next.js

### Deployment (Optional)
- Vercel

---

## Data Source
- Historical match data scraped from **fir.tournamentsoftware.com**

---

## Project Goals

### 1. Data Collection
- Scrape historical racketlon match data
- Store raw match, player, and tournament information
- Handle inconsistencies, missing values, and formatting issues

### 2. Data Cleaning & Feature Engineering
- Identify meaningful performance indicators, such as:
  - Discipline-specific results
  - Score margins
  - Win/loss ratios
  - Recent form
  - Head-to-head history
- Normalize and structure data for modeling
- Evaluate feature importance and impact on predictions

### 3. Player Rating System
- Design a rating system based on historical match outcomes
- Ratings should:
  - Reflect overall and discipline-specific skill
  - Update over time as new matches are played
  - Serve as core input features for prediction models

### 4. Match Outcome Prediction
- Frame the task as a supervised learning problem
- Train and evaluate models to predict match outcomes
- Compare baseline and advanced models
- Balance predictive accuracy with interpretability

### 5. Web Interface (Stretch Goal)
- Build a simple web application to:
  - View predicted match outcomes
  - Compare players and ratings
- Deploy the application using Vercel

---

## Milestones & Timeline 2026

### January – February
- Scrape historical match data
- Build an initial data storage pipeline

### February – March
- Clean and preprocess data
- Explore and engineer relevant features

### March – April
- Implement player rating system
- Train and evaluate prediction models

### April – May
- Finalize model and performance evaluation
- *(If time permits)* Build and deploy the web interface

---

## Expected Outcome
By the end of this project, the system should be able to:
- Assign meaningful ratings to racketlon players
- Predict match outcomes with reasonable accuracy
- Provide a solid foundation for future improvements and extensions
