# Race and Gender in the Workforce 

## Project Overview

American companies have been engaged in equal opportunity and diversity efforts since the passage of the Civil Rights Act in 1964.  The purpose of this project was to expore available data from the Bureau of Labor Statistics to see how women and racial/ethnic minorities are faring 55 years later.  I was particularly interested in the status of these groups in higher level jobs, defined for the purpose as professionals, managers, and chief executives, and in their progress over the period between 2010 and 2019 (the latest year for which BLS had data.  This was a decade in which many companies' diversity efforts were seen to mature and in which many made considerable investments. I also proposed to project whatever trends I found into the future in order to estimate where workforce diversity might be expected to stand ten years hence.

## Methodology

I started with a csv file downloaded from the BLS website.  In truth, much of the arithmetic involved in calculating representation and change over time may have been more easily accomplished in Excel, but I wanted to challenge myself to execute it python.  To that end, I built various functions to convert the data into dataframes, clean and order them, perform the calculations, and plot results.

The basic data--change in a group's representation over time--is useful to see, but not very informative in itself.  For example, this chart shows us that African-American's share of each of the job categories went up in the past ten years.  

![Sample Graph](https://github.com/mfineman/Race-Gender-Analysis/blob/main/Images/AArep.png)

However, the real story is not in the lines, but between them.  While their share of each job level increased, so did their share of all jobs.  So how much was their increase in managerial jobs, say, true gain and how much was just that African-Americans make up more of the workforce than ten years ago?

To gauge "real" advancement, I added a measure I'm calling *advancement ratio*.  Advancement ratio shows a group's share of a category of jobs compared to (that is, divided by) its share of the total workforce.  A ratio of 100% would mean that a group's representation in a category is proportional to its representation in the workforce.  A ratio of under 100% means the group is under-represented; a ratio of higher than 100% means the group is over-represented in that set of jobs.  These ratios give a clearer picture of how much a group is actually progressing, as in the examples below:

![Twenty year advancement ratios, Africa-Americans](https://github.com/mfineman/Race-Gender-Analysis/blob/main/Images/African-Americans_ratios_2010-2029.png)

![Sample Graph](https://github.com/mfineman/Race-Gender-Analysis/blob/main/Images/African-Americans_ratio_change19.png)

(In its diversity practice, a given organization could use advancement ratios to trace a demographic group's progress from feeder jobs to the next level and identify the points along a career path where progress tends to stall.  In the case of national BLS statistics, however, we don't have fine enough job categories to map out distinct career paths.)

You will notice that the above chart of advancement ratios projects out to 2029.  To project these estimates, I figured the average annual rate of change in representation for each demographic group in each job category and applied that rate to each of the next ten years. This yielded a new set of data from which projected advancement ratios and change in the ratios could be determined.

## View Results

A website displaying the most pertinent visualizations, along with observations about their significance, can be seen at https://mfineman.github.io/Race-Gender-Analysis/.  Additional visualizations, the original dataset, and my code are available in this repository.

