# Hotel Booking Data Analysis

Welcome to the Hotel Booking Data Analysis project! This project utilizes Streamlit to create an interactive web application for analyzing hotel booking data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Exercises](#exercises)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Visualization and Analysis](#visualization-and-analysis)

## Introduction

This project focuses on analyzing hotel booking data through various visualizations and statistical methods. The data is cleaned, preprocessed, and then used for exploratory data analysis (EDA) and machine learning tasks such as PCA and K-Nearest Neighbors classification.

## Features

- Data loading and caching
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Visualization of various data insights
- Principal Component Analysis (PCA)
- K-Nearest Neighbors (KNN) classification
- Interactive visualizations with Streamlit
- Exercises to enhance understanding of data visualization principles

## Setup and Installation

To run this project locally, follow these steps:

1. Clone the repository:
   git clone https://github.com/Megavarthini2203/dev-app.git
   
2. Navigate to the project directory:
   cd hotel-booking-data-analysis

3. Install the required Packages:
   pip install -r requirements.txt

 4.Run the Streamlit Application:
  streamlit run app.py

## Usage

Once the application is running, you can navigate through different sections using the sidebar. Select different exercises to explore various aspects of data visualization and analysis.

## Exercises

The application includes several exercises to help understand different aspects of data visualization and analysis:

- Exercise 1: Choosing the right chart types
- Exercise 2: Arranging elements for clarity
- Exercise 3: Employing color schemes strategically
- Exercise 4: Incorporating interactivity
- Exercise 5: Conduct exploratory data analysis using visualization
- Exercise 6: Craft visual presentations of data for effective communication
- Exercise 7: Use knowledge of perception and cognition to evaluate visualization design alternatives
- Exercise 8: Design and evaluate color palettes for visualization based on principles of perception
- Exercise 9: Apply data transformations such as aggregation and filtering for visualization

## Data Cleaning and Preprocessing
The data is cleaned and preprocessed before analysis:

- Fill missing values for children, agent, and country
- Remove rows with zero guests
- Convert categorical columns to string type
- Handle date parsing
- Apply label encoding to categorical features
- Remove outliers using binning
- Fill remaining NaN values with the mean

## Visualization and Analysis
Various visualizations are included to explore the data:

- Distribution plots
- Box plots
- Scatter plots
- Heatmaps
- 3D scatter plots
- PCA visualization
- KNN decision regions
