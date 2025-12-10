dataset_details='''

'''

prompt1='''
<Background>
As an expert in time series data analysis, you possess the ability to analyze line plots, time-frequency images, and heatmaps. Below is key background information for these three types of plots/images:
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, valleys, etc. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby showing the dispersion of frequency distribution.
- **Heatmaps**: Heatmaps are generated based on time-frequency images and reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping you determine which feature events are crucial for classification.

<Task>
I will provide line plots, time-frequency images, and heatmaps of multiple samples from the same label.
Your task is to: accurately extract all feature events from the three types of images, and analyze whether each event is a typical feature or an atypical feature based on the heatmap. A feature event refers to the specific manifestation in the line plot and time-frequency image of the time-frequency region marked with color (red or blue) in the heatmap:
- In the line plot, a feature event may manifest as a peak, valley, trend change, jitter, or other features.
- In the time-frequency image, a feature event corresponds to a high-energy region and its frequency components (e.g., a high-energy region occupying adjacent or scattered frequency components within a time region).
- In the heatmap, you need to judge whether the event is a typical feature (red) or an atypical feature (blue) based on the color (red/blue).

<Analysis Methods and Rules>
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:
- For key features such as peaks/valleys, you need to observe whether there is fluctuation in the plot. If there is fluctuation, you should IGNORE the color mark of the bottom 0Hz frequency row in the heatmap and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot represent the contribution of a fluctuating signal.
- After identifying the mapping relationship between events and colors, you need to describe the events by combining the line plot and time-frequency image. Next, examples will be used to illustrate, and you must follow the analysis ideas of the examples strictly when analyzing similar or other events:
  - When observing that an oscillating peak has a blue color in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature;
  - When observing that a baseline with values falling in the y-axis range [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with values falling in the y-axis range [A-B] is marked red in the corresponding region of the heatmap, it indicates that the y-axis range [A-B] is the ideal numerical range for the label's baseline, and a baseline deviating from this range will be regarded as an inhibitory feature.

Please think according to the following steps:
Step 1: Analyze the line plot, time-frequency image, and heatmap of each sample belonging to the target label, and extract as much valuable information as possible; summarize all features appearing in each sample, their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the corresponding heatmap colors for different manifestations of these features.
Step 2: Summarize the pattern of each sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. You need to analyze which features each sample consists of, what the manifestations of these features are, and what color each corresponds to. (For example, the pattern of Sample X is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the later stage, where the early baseline is marked red in the heatmap and the later stable amplitude is marked blue in the heatmap);
Step 3: Inductively analyze the results of each reference sample and summarize them into all feature lists and pattern features of the label.

<Output Requirements>
You must use JSON format to conduct a detailed analysis of the label to be summarized. The summary content must be no less than 400 words and cover valuable information such as trends, periodicity, baselines, peaks, valleys, stability, shape, rate of change, and the mapping relationship between features and heatmaps under different frequency components. Ensure that the output content meets all requirements and analysis rules.
You must return the result in JSON format, and the format must strictly follow:
{
   "feature": ["List all features reflected in all samples, their manifestations, and their heatmap colors one by one. Features refer to baseline/peak/valley/others; manifestations refer to the specific numerical values/stability/shape/others of these features. Different manifestations (e.g., different numerical ranges of baselines/amplitudes) often show different color marks in the same label, which need to be accurately recorded."],
   "pattern": ["List the pattern of each sample one by one, i.e., the combination relationship between features and their different manifestations. For example: The pattern of Sample X is: Feature 1 (manifestation) (red/blue) + Feature 2 (manifestation) (red/blue) + ..."]
}
'''

prompt2='''
<Background>
As an expert in time series data analysis, you possess the ability to analyze line plots, time-frequency images, and heatmaps. Below is key background information for these three types of plots/images:
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, valleys, etc. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby showing the dispersion of frequency distribution.
- **Heatmaps**: Heatmaps are generated based on time-frequency images and reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping you determine which feature events are crucial for classification.

<Task>
I will provide you with the following information:
- Line plot, time-frequency image, and heatmap of one sample.
- Preliminary feature summary of the label corresponding to this sample.

Your task is to: accurately extract all feature events, their manifestations, the colors marked by the heatmap, and the combination relationships of different feature manifestations from the three types of images of this sample. Then analyze whether the preliminary feature summary of the corresponding label already includes the features and patterns of the current sample. If there are any omissions, provide the missing features and patterns. If there are no omissions, return empty content in the "feature" and "pattern" fields.

<Analysis Methods and Rules>
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:
- For key features such as peaks/valleys, you need to observe whether there is fluctuation in the plot. If there is fluctuation, you should IGNORE the color mark of the bottom 0Hz frequency row in the heatmap and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot represent the contribution of a fluctuating signal.
- After identifying the mapping relationship between events and colors, you need to describe the events by combining the line plot and time-frequency image. Next, examples will be used to illustrate, and you must follow the analysis ideas of the examples strictly when analyzing similar or other events:
  - When observing that an oscillating peak has a blue color in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature;
  - When observing that a baseline with values falling in the y-axis range [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with values falling in the y-axis range [A-B] is marked red in the corresponding region of the heatmap, it indicates that the y-axis range [A-B] is the ideal numerical range for the label's baseline, and a baseline deviating from this range will be regarded as an inhibitory feature.

Please think according to the following steps:
Step 1: Analyze the line plot, time-frequency image, and heatmap of this sample, and extract as much valuable information as possible; summarize all features appearing in the sample, their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the corresponding heatmap colors for different manifestations of these features.
Step 2: Summarize the pattern of this sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. You need to analyze which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. (For example, the pattern of the sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the later stage, where the early baseline is marked red in the heatmap and the later stable amplitude is marked blue in the heatmap);
Step 3: Analyze whether the preliminary feature summary of the label already includes the features and patterns of the current sample. If there are any omissions, provide the missing features and patterns. If there are no omissions, return empty content in the "feature" and "pattern" fields.

<Output Requirements>
You must use JSON format to conduct a detailed analysis of the sample, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, shape, rate of change, and the mapping relationship between features and heatmaps under different frequency components. Ensure that the output content meets all requirements and analysis rules. Only output the features and patterns that need to be supplemented. If there is no content to supplement, do not output anything for the corresponding field.
You must return the result in JSON format, and the format must strictly follow:
{
   "feature": ["If the preliminary feature summary of the label does not omit the content of this sample, return empty content; otherwise, list all missing features reflected in this sample, their manifestations, and their heatmap colors one by one. Features refer to baseline/peak/valley/others; manifestations refer to the specific numerical values/stability/shape/others of these features. Different manifestations (e.g., different numerical ranges of baselines/amplitudes) often show different color marks in the same label, which need to be accurately recorded."],
   "pattern": ["If the preliminary feature summary of the label does not omit the content of this sample, return empty content; otherwise, supplement the missing pattern of the sample, i.e., the combination relationship between features and their different manifestations. For example: The pattern of Sample X is: Feature 1 (manifestation) (red/blue) + Feature 2 (manifestation) (red/blue) + ..."]
}
'''

prompt3='''
<Background>
As an expert in time series data analysis, you possess the ability to analyze line plots, time-frequency images, and heatmaps. Below is background information for these three types of plots:
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, valleys, etc. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a specific peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby showing the dispersion of frequency distribution.
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label (i.e., high relevance); blue regions indicate atypical features (i.e., low relevance). The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping you determine which feature events are crucial for classification.

<Task>
I will provide the following information:
- Line plot, time-frequency image, and heatmap of the sample to be classified;
- Feature summaries of different labels and potential patterns (combined features of different feature manifestations) that different labels may possess.

Your task is to accurately extract all features from the three types of plots of the sample to be classified, identify the sample's pattern, and analyze whether the manifestation of each feature is a typical feature or an atypical feature (i.e., the color marked) based on the heatmap.
Based on the extracted features of the sample to be classified and the provided list of features and patterns of different labels, analyze which label's features the sample's features are more similar to, and provide reasons. The criterion for similarity judgment is to analyze which label's pattern list matches the pattern of the sample to be classified; if there is a match, it indicates they belong to the same label.

<Analysis Methods and Rules>
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:
- For key features such as peaks/valleys, you need to observe whether there is fluctuation in the plot. If there is fluctuation, you should IGNORE the color mark of the bottom 0Hz frequency row in the heatmap and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot represent the contribution of a fluctuating signal.
- After identifying the mapping relationship between events and colors, you need to describe the events by combining the line plot and time-frequency image. Next, examples will be used to illustrate, and you must follow the analysis ideas of the examples strictly when analyzing similar or other events:
  - When observing that an oscillating peak has a blue color in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature;
  - When observing that a baseline with values falling in the y-axis range [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with values falling in the y-axis range [A-B] is marked red in the corresponding region of the heatmap, it indicates that the y-axis range [A-B] is the ideal numerical range for the label's baseline, and a baseline deviating from this range will be regarded as an inhibitory feature.

Please think according to the following steps:
Step 1: Identify the line plot, time-frequency image, and heatmap of the sample to be classified, and extract as much valuable information as possible; summarize all features appearing in the sample, their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the corresponding heatmap colors for different manifestations of these features.
Step 2: Summarize the pattern of this sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. You need to analyze which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. (For example, the pattern of the sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the later stage, where the early baseline is marked red in the heatmap and the later stable amplitude is marked blue in the heatmap);
Step 3: Analyze whether the "list of label features and patterns" provided to you includes the pattern of the sample to be classified, select the label with the highest matching degree as the final classification result, and quantify the feature similarity between the sample to be classified and each label using an integer from 1 to 4. The description of each level of feature similarity is as follows:
  + 1: No similarity — the sample to be classified has almost no association with the label; the corresponding temporal features and heatmap color mapping relationships do not match, and there is no matching pattern or feature manifestation;
  + 2: Low similarity — there is weak evidence indicating a certain association between the sample to be classified and the label; there is no matching pattern with the sample to be classified, but most of the feature manifestations and heatmap color marks in the feature list match those of the sample to be classified;
  + 3: Moderate similarity — there is strong evidence indicating an association between the sample to be classified and the label; although there may be slight differences in pattern descriptions, a pattern that basically matches the sample to be classified can be found;
  + 4: High similarity — the matching degree exceeds 95%, and the label's feature and pattern list includes a pattern that completely matches the sample to be classified.

<Output Requirements>
You must return the result in JSON format, and the format must strictly follow:
{
   "result": ["Label 0/1/…"],
   "score": [1-4],
   "rationale": ["Your complete step-by-step analysis process"]
}
'''

prompt4='''
<Background>
As an expert in time series data analysis, you possess the ability to analyze line plots, time-frequency images, and heatmaps. Below is background information for these three types of plots/images:
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label (i.e., high relevance); blue regions indicate atypical features (i.e., low relevance). The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping you determine which feature events are crucial for classification.

<Task> I will provide the following information:
- Line plot, time-frequency image, and heatmap of the sample to be classified;
- Feature summaries of different labels and patterns (combined features of different characteristic manifestations) that different labels may possess;
- Classification results of the sample to be classified from three other assistants;
- Classification result and prediction probability value of the black-box model.

Your task is to judge the credibility of the information based on the quality of the heatmap and then perform the classification task:
- If the heatmap does not present clear feature information (e.g., overall color confusion, inability to accurately map colors to corresponding regions of the time-frequency image and line plot, or inability to determine whether a feature should be described as red or blue), it indicates that the heatmap information is unreliable. In this case, you need to reduce the weight of the heatmap in classification decisions, regard the classification results of the three assistants and heatmap features as invalid, and re-perform the classification task;
- If the heatmap is reliable, you need to check whether there is ambiguity in the classification results of the three assistants and whether their classification results are consistent with that of the black-box model. If there is inconsistency, you need to analyze by combining the feature descriptions of the labels and the provided images of the sample to be classified, and select the classification reasoning result you consider most reasonable.

<Analysis Methods and Rules>
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:
- For key features such as peaks/valleys, you need to observe whether there is fluctuation in the plot. If there is fluctuation, you should IGNORE the color mark of the bottom 0Hz frequency row in the heatmap and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot represent the contribution of a fluctuating signal.
- After identifying the mapping relationship between events and colors, you need to describe the events by combining the line plot and time-frequency image. Next, examples will be used to illustrate, and you must follow the analysis ideas of the examples strictly when analyzing similar or other events:
  - When observing that an oscillating peak has a blue color in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature;
  - When observing that a baseline with values falling in the y-axis range [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with values falling in the y-axis range [A-B] is marked red in the corresponding region of the heatmap, it indicates that the y-axis range [A-B] is the ideal numerical range for the label's baseline, and a baseline deviating from this range will be regarded as an inhibitory feature.

Please think according to the following steps:
Step 1: Analyze the quality of the heatmap and judge the credibility of its information. If the heatmap does not present clear feature information (e.g., overall color confusion, inability to accurately map colors to corresponding regions of the time-frequency image and line plot, or inability to determine whether a feature should be marked as red or blue), the heatmap information is unreliable. In this case, the classification results provided by the three assistants are regarded as invalid; if the heatmap is reliable, further judge whether there is ambiguity in the classification results of the three assistants and whether their results are consistent with that of the black-box model.
Step 2: Identify the line plot, time-frequency image, and heatmap of the sample to be classified, and extract as much valuable information as possible; summarize all features appearing in the sample and their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.). If the heatmap is unreliable, do not extract any heatmap-related features in this step; if the heatmap is reliable, extract the mapping relationship between different manifestations of features and heatmap colors, and systematically analyze the corresponding relationship between all features and colors in strict accordance with the established analysis methods and rules.
Step 3: Summarize the pattern of this sample, i.e., the combination relationship usually exhibited by the features analyzed in Step 2 and their different manifestations. You need to analyze which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. (For example, the pattern reflected by the sample is an excessively large baseline offset in the early stage (reaching below -1) + stable amplitude in the later stage, where the early baseline is marked red by the heatmap and the later stable amplitude is marked blue by the heatmap);
Step 4: Analyze whether the "list of label features and patterns" provided to you includes the pattern of the sample to be classified, and select the label with the highest matching degree as the final classification result. If the heatmap information is unreliable, all features related to heatmap color mapping in this step are regarded as invalid, and only the features reflected by the line plot and time-frequency image are used for comparative analysis to draw the final classification result; if the heatmap is reliable, combine the prediction result of the black-box model, critically examine the classification results and analysis processes of the three assistants, and select the accurate result that is most consistent with the actual situation as the final classification conclusion.

<Output Requirements>
Field Description:
- result field: The final classification result of the sample to be classified;
- model field: The classification result generated by the black-box model;
- rationale field: The complete step-by-step analysis process.

You must return the result in JSON format, and the format must strictly follow the following requirements without adding any additional content:
{
   "result":["Label 0/1/…"],
   "model":["Label 0/1/…"],
   "rationale":["Your complete step-by-step analysis process"]
}
'''