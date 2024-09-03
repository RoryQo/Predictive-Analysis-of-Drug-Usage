# R-Supervised-Learning-Lab
#### Purpose
The purpose of lab (1) is to predict hard drug use with supervised learning models including random forests, and naive bayes model; using false positive rates, true positive rates, true negative rates and accuracy rates to assess the usefulness and validity of the models.

The purpose lab (2) is to use principal components to explain proportions of variance between countries using the unsupervised learning model Principal component analysis
#### Notes
+ Code(1) and (2) are original rmd files 
+ markdown (1) and (2) are the knitted r files output for github
+ Code(1) and markdown(1) are the same code and code(2) and markdown(2) are the same code

Code(1) uses national health data to predict hard drug use from other health markers
### NHANES data (available directly in r)
#### Target
+ `HardDrugs` Participant has tried cocaine, crack cocaine, heroin or methamphetamine. Reported
for participants aged 18 to 69 years as Yes or No.
#### Features
+ `BMI` Body mass index (weight/height2 in kg/m2). Reported for participants aged 2 years or older.
+ `Gender` Gender (sex) of study participant coded as male or female
+ `Age` Age in years at screening of study participant. Note: Subjects 80 years or older were recorded
as 80.
+ `Race1` Reported race of study participant: Mexican, Hispanic, White, Black, or Other.
+ `Education` Educational level of study participant Reported for participants aged 20 years or older.
One of 8thGrade, 9-11thGrade, HighSchool, SomeCollege, or CollegeGrad.
+ `MaritalStatus` Marital status of study participant. Reported for participants aged 20 years or older.
One of Married, Widowed, Divorced, Separated, NeverMarried, or LivePartner (living
with partner).
+ `HHIncome` Total annual gross income for the household in US dollars. One of 0 - 4999, 5000
,9,999, 10000 - 14999, 15000 - 19999, 20000 - 24,999, 25000 - 34999, 35000 - 44999,
45000 - 54999, 55000 - 64999, 65000 - 74999, 75000 - 99999, or 100000 or More.
+ `HomeOwn` One of Home, Rent, or Other indicating whether the home of study participant or someone in their family is owned, rented or occupied by some other arrangment
+ `Weight` Weight in kg
+ `Height` Standing height in cm. Reported for participants aged 2 years or older.
+ `Pulse` 60 second pulse rate
+ `BPSysAve` Combined systolic blood pressure reading, following the procedure outlined for BPXSAR
+ `BPDiaAve` Combined diastolic blood pressure reading, following the procedure outlined for BPXDAR.
+ `Diabetes` Study participant told by a doctor or health professional that they have diabetes. Reported
for participants aged 1 year or older as Yes or No.
+ `HealthGen` Self-reported rating of participant’s health in general Reported for participants aged 12
years or older. One of Excellent, Vgood, Good, Fair, or Poor.
+ `DaysPhysHlthBad` Self-reported number of days participant’s physical health was not good out of
the past 30 days. Reported for participants aged 12 years or older.
+ `DaysMentHlthBad` Self-reported number of days participant’s mental health was not good out of
the past 30 days. Reported for participants aged 12 years or older
+ **Depressed** Self-reported number of days where participant felt down, depressed or hopeless. Reported for participants aged 18 years or older. One of None, Several, Majority (more than
half the days), or AlmostAll.
+ `SleepHrsNight` Self-reported number of hours study participant usually gets at night on weekdays
or workdays. Reported for participants aged 16 years and older.
+ `SleepTrouble` Participant has told a doctor or other health professional that they had trouble sleeping. Reported for participants aged 16 years and older. Coded as Yes or No.
+ `Smoke100` Study participant has smoked at least 100 cigarettes in their entire life. Reported for
participants aged 20 years or older as Yes or No
+ `Marijuana` Participant has tried marijuana. Reported for participants aged 18 to 59 years as Yes
or No. AgeFirstMarijAge participant first tried marijuana. Reported for participants aged 18
to 59 years
+ `Smoke100` Study participant has smoked at least 100 cigarettes in their entire life. Reported for
participants aged 20 years or older as Yes or No
+ `AlcoholDay Average` number of drinks consumed on days that participant drank alcoholic beverages. Reported for participants aged 18 years or older.





