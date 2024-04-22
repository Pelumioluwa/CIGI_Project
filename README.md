# Project SentiScribe

Project Senticribe aims to analyze video discussions from the UN Open-ended Working Group (OEWG) by analyzing the video recordings of meetings across all sessions. This project involves the development of four models to provide insights into the discussions and dynamics of the meetings, overall conclusions and sentiments of speakers .

## Data
This project utilized video data. A sample can be found [here](https://webtv.un.org/en/asset/k1a/k1a35z9guj).

## Models

### 1. Space Chat (Retrieval Augmented Generation)
Space Chat utilizes Retrieval Augmented Generation (RAG), leveraging OpenAI and AI search as a vector base. It enables users to easily retrieve information about the meetings, summarize discussions, and gain insights. The model allows users to interactively query the meetings and obtain relevant information.

### 2. Language Analysis Model
The Language Analysis Model assigns various countries to different clusters based on the language used in their speeches. It highlights the most frequently used words by different speakers and identifies clusters of speakers who use similar language patterns. This model takes speeches from different countries in the UN OEWG meetings and analyzes the words commonly used by speakers.

### 3. Sentiment Analysis Model
The Sentiment Analysis Model assigns various countries to different clusters based on their sentiments expressed during the meetings. Unlike traditional sentiment analysis (positive/negative), this model provides contextual clustering based on countries sharing similar sentiments or focus areas. It identifies clusters of countries with similar sentiments or thematic focus.

### 4. Topic Modelling
The Topic Modelling Model assigns various countries to different clusters based on the topics of focus in their discussions. It utilizes OpenAI to provide deeper insights into the topics discussed during the meetings. This model identifies clusters of countries focusing on similar topics and extracts key themes from their discussions.

## Visualization

Language Analysis Visualization
To visualize the language analysis of the countries, you can access the interactive visualization on Tableau Public. Click the link below:

[UNOEWG Language Analysis Clusters](https://public.tableau.com/app/profile/pelumi.abiola/viz/UNOEWGLanguageAnalysisClusters/LanguageAnalysis)

Sentiment Analysis Visualization
To visualize the sentiment analysis of the countries, you can access the interactive visualization on Tableau Public. Click the link below:

[UNOEWG Language Analysis Clusters](https://public.tableau.com/app/profile/pelumi.abiola/viz/UNOEWGSentimentAnalysisClusters/Sentimentanalysis)


## Contributions
Contributions to Project Senticribe are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it according to the terms of the license.
