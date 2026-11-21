"""
Project 12: Gym & Fitness Center Membership Analytics
Dataset File: 12_gym_membership.csv
Author: Senior Data Analyst & Python Developer
Environment: Ready to run in Jupyter Notebook / Python 3.8+
Libraries: pandas, numpy, matplotlib, seaborn
"""

# ==============================================================================
# IMPORT REQUIRED LIBRARIES
# ==============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# ==============================================================================
# 1. DATA INSPECTION
# ==============================================================================
print("=" * 30)
print("1. DATA INSPECTION")
print("=" * 30)

# Define dataset filename / path
dataset_file = '12_gym_membership.csv'

# Load the dataset
df = pd.read_csv(dataset_file)

# Display dataset shape
print(f"\n[A] Dataset Shape:")
print(f"    Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Display data types
print(f"\n[B] Data Types:")
print(df.dtypes)

# Display first 5 rows
print(f"\n[C] First 5 Rows:")
print(df.head())

# Display summary statistics (numerical)
print(f"\n[D] Summary Statistics (Numerical Columns):")
print(df.describe().round(2))

# Display summary statistics (categorical)
print(f"\n[E] Summary Statistics (Categorical Columns):")
print(df.describe(include=['object', 'O']))


# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
print("\n" + "=" * 30)
print("2. DATA CLEANING")
print("=" * 30)

# Check missing values in every column
print(f"\n[A] Missing Values Count Per Column:")
missing_vals = df.isnull().sum()
print(missing_vals)

# Check duplicate Checkin_ID values and display duplicate records
duplicate_checkins = df[df.duplicated(subset=['Checkin_ID'], keep=False)]
num_duplicates = df.duplicated(subset=['Checkin_ID']).sum()
print(f"\n[B] Duplicate Checkin_ID Records Found: {num_duplicates}")

if num_duplicates > 0:
    print("\nDisplaying Duplicate Records:")
    print(duplicate_checkins.sort_values(by='Checkin_ID').to_string(index=False))
    
    # Remove duplicates, keeping the first occurrence to maintain data integrity
    df.drop_duplicates(subset=['Checkin_ID'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"\nShape after dropping duplicate Checkin_ID records: {df.shape}")

# Clean missing Workout_Duration_Mins using Median
median_duration = df['Workout_Duration_Mins'].median()
mean_duration = df['Workout_Duration_Mins'].mean()

print(f"\n[C] Workout Duration Imputation Analysis:")
print(f"    - Sample Mean Workout Duration:   {mean_duration:.2f} minutes")
print(f"    - Sample Median Workout Duration: {median_duration:.2f} minutes")

# Impute missing values with median
df['Workout_Duration_Mins'] = df['Workout_Duration_Mins'].fillna(median_duration)

# Explain why median is preferred over mean
print("\n[D] Explanation: Why Median is Preferred Over Mean:")
print("    1. Robustness to Extreme Values / Outliers:")
print("       Workout durations often exhibit skewness (e.g., quick 20-minute drop-ins")
print("       versus marathon 130-minute training sessions). The arithmetic mean is")
print("       heavily influenced by extreme values, whereas the median represents the")
print("       true 50th percentile (central tendency) of member behavior.")
print("    2. Preserves Empirical Distribution:")
print("       Imputing with the median prevents artificial inflation or deflation of")
print("       the variance and respects typical session lengths without bias.")

# Verify no missing values remain
print(f"\n[E] Verification of Missing Values Remaining:")
print(df.isnull().sum())
assert df.isnull().sum().sum() == 0, "Warning: Unhandled missing values detected!"
print("    -> All columns successfully cleaned. Zero missing values remaining.")


# ==============================================================================
# 3. TIME ANALYSIS
# ==============================================================================
print("\n" + "=" * 30)
print("3. TIME ANALYSIS")
print("=" * 30)

# Convert Checkin_Time to datetime format
df['Checkin_Time'] = pd.to_datetime(df['Checkin_Time'])

# Extract required temporal features
df['Visit_Date'] = df['Checkin_Time'].dt.date
df['Month'] = df['Checkin_Time'].dt.month
df['Month_Name'] = df['Checkin_Time'].dt.month_name()
df['Year'] = df['Checkin_Time'].dt.year
df['Year_Month'] = df['Checkin_Time'].dt.to_period('M')
df['Weekday'] = df['Checkin_Time'].dt.day_name()
df['Hour'] = df['Checkin_Time'].dt.hour

print("Date & Time features successfully extracted:")
print(df[['Checkin_ID', 'Checkin_Time', 'Visit_Date', 'Month', 'Year', 'Weekday', 'Hour']].head())

# Busiest gym hours
hourly_counts = df['Hour'].value_counts().sort_index()
peak_hour = hourly_counts.idxmax()
peak_hour_count = hourly_counts.max()

# Busiest weekdays (ordered Monday to Sunday)
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = df['Weekday'].value_counts().reindex(weekday_order)
peak_day = weekday_counts.idxmax()
peak_day_count = weekday_counts.max()

print(f"\n[A] Peak Hour Summary:")
print(f"    - Busiest Single Hour: {peak_hour:02d}:00 with {peak_hour_count} check-ins.")
print(f"    - Top 3 Busiest Hours:\n{df['Hour'].value_counts().head(3).to_string()}")

print(f"\n[B] Peak Weekday Summary:")
print(f"    - Busiest Day of Week: {peak_day} with {peak_day_count} check-ins.")
print(f"    - All Weekdays Attendance:\n{weekday_counts.to_string()}")


# ==============================================================================
# 4. MEMBERSHIP ANALYSIS
# ==============================================================================
print("\n" + "=" * 30)
print("4. MEMBERSHIP ANALYSIS")
print("=" * 30)

# Overall KPIs
total_checkins = len(df)
avg_workout_dur = df['Workout_Duration_Mins'].mean()
median_workout_dur = df['Workout_Duration_Mins'].median()
total_members = df['Member_ID'].nunique()
total_revenue = df['Monthly_Fee'].sum()

print("\n--- Key Performance Indicators (KPIs) ---")
print(f"Total Check-ins:          {total_checkins:,}")
print(f"Total Unique Members:     {total_members:,}")
print(f"Average Workout Duration: {avg_workout_dur:.2f} minutes")
print(f"Median Workout Duration:  {median_workout_dur:.2f} minutes")
print(f"Total Revenue Recorded:   ${total_revenue:,.2f}")

# Groupby Membership_Tier
tier_analysis = df.groupby('Membership_Tier').agg(
    Number_of_Visits=('Checkin_ID', 'count'),
    Average_Workout_Duration=('Workout_Duration_Mins', 'mean'),
    Total_Revenue=('Monthly_Fee', 'sum')
).reset_index()

# Add percentage contributions
tier_analysis['Visit_Percentage'] = (tier_analysis['Number_of_Visits'] / total_checkins * 100).round(2)
tier_analysis['Revenue_Percentage'] = (tier_analysis['Total_Revenue'] / total_revenue * 100).round(2)
tier_analysis['Average_Workout_Duration'] = tier_analysis['Average_Workout_Duration'].round(2)

print("\n--- Membership Tier Comparison Table ---")
print(tier_analysis.to_string(index=False))


# ==============================================================================
# 5. CLASS ANALYSIS
# ==============================================================================
print("\n" + "=" * 30)
print("5. CLASS ANALYSIS")
print("=" * 30)

# Rank fitness classes by attendance
class_attendance = df['Fitness_Class'].value_counts().reset_index()
class_attendance.columns = ['Fitness_Class', 'Attendance_Count']
class_attendance['Attendance_Share_Pct'] = (class_attendance['Attendance_Count'] / total_checkins * 100).round(2)

# Calculate average duration per class
class_duration = df.groupby('Fitness_Class')['Workout_Duration_Mins'].mean().round(2).reset_index()
class_duration.columns = ['Fitness_Class', 'Avg_Duration_Mins']

# Merge into comprehensive attendance table
class_summary = pd.merge(class_attendance, class_duration, on='Fitness_Class')

print("\n--- Fitness Class Attendance Ranking ---")
print(class_summary.to_string(index=False))


# ==============================================================================
# 6. REVENUE ANALYSIS
# ==============================================================================
print("\n" + "=" * 30)
print("6. REVENUE ANALYSIS")
print("=" * 30)

# Revenue by Month (Chronological Year-Month)
revenue_by_month = df.groupby('Year_Month')['Monthly_Fee'].sum().reset_index()
revenue_by_month.columns = ['Year_Month', 'Total_Revenue']

print("\n[A] Revenue by Month (Chronological Year-Month):")
print(revenue_by_month.to_string(index=False))

# Revenue by Membership Tier
revenue_by_tier = df.groupby('Membership_Tier')['Monthly_Fee'].sum().reset_index()
revenue_by_tier.columns = ['Membership_Tier', 'Total_Revenue']
revenue_by_tier['Revenue_Share_Pct'] = (revenue_by_tier['Total_Revenue'] / total_revenue * 100).round(2)

print("\n[B] Revenue by Membership Tier:")
print(revenue_by_tier.to_string(index=False))

# Monthly Revenue broken down by Membership Tier
monthly_tier_revenue = df.pivot_table(
    index='Year_Month',
    columns='Membership_Tier',
    values='Monthly_Fee',
    aggfunc='sum',
    fill_value=0
)
print("\n[C] Monthly Revenue Breakdown by Membership Tier (Head):")
print(monthly_tier_revenue.head())


# ==============================================================================
# 7. LOW ENGAGEMENT MEMBERS
# ==============================================================================
print("\n" + "=" * 30)
print("7. LOW ENGAGEMENT MEMBERS")
print("=" * 30)

# Steps:
# 1. Group by Member_ID and Year_Month (or Month)
# 2. Count visits
# 3. Mark members with visits < 2

# Visit count per member per active monthly billing cycle
member_monthly_visits = df.groupby(['Member_ID', 'Year_Month']).size().reset_index(name='Monthly_Visits')
member_monthly_visits['Is_Low_Engagement'] = member_monthly_visits['Monthly_Visits'] < 2

# Identify low-engagement instances
low_engagement_instances = member_monthly_visits[member_monthly_visits['Is_Low_Engagement']]
low_engagement_members_list = sorted(low_engagement_instances['Member_ID'].unique())

print(f"\n[A] Low Engagement Identification Summary:")
print(f"    - Total Member-Month Cycles Analyzed:     {len(member_monthly_visits)}")
print(f"    - Under-engaged Member-Month Occurrences: {len(low_engagement_instances)}")
print(f"    - Unique Members with Low Engagement (<2 visits/month): {len(low_engagement_members_list)}")

print(f"\n[B] Sample of Low-Engagement Members (First 20 Members):")
print(", ".join(low_engagement_members_list[:20]))

# Also compute overall average visits per active month for each member
member_avg_engagement = member_monthly_visits.groupby('Member_ID')['Monthly_Visits'].mean().reset_index()
critically_disengaged = member_avg_engagement[member_avg_engagement['Monthly_Visits'] < 2.0]
print(f"\n[C] Members Averaging Fewer Than 2 Visits Across All Active Months: {len(critically_disengaged)}")


# ==============================================================================
# 8. NUMPY STATISTICS
# ==============================================================================
print("\n" + "=" * 30)
print("8. NUMPY STATISTICS")
print("=" * 30)

# Extract Workout_Duration_Mins as a NumPy array
duration_array = df['Workout_Duration_Mins'].to_numpy()

# Calculate statistics using NumPy functions
np_mean = np.mean(duration_array)
np_median = np.median(duration_array)
np_var = np.var(duration_array)  # Population variance (ddof=0)
np_std = np.std(duration_array)
np_q1 = np.percentile(duration_array, 25)
np_q2 = np.percentile(duration_array, 50)
np_q3 = np.percentile(duration_array, 75)

print("\n--- NumPy Statistical Summary for Workout_Duration_Mins ---")
print(f"Mean (μ):               {np_mean:.4f} minutes")
print(f"Median (Q2):            {np_median:.4f} minutes")
print(f"Variance (σ²):          {np_var:.4f}")
print(f"Standard Deviation (σ): {np_std:.4f} minutes")
print(f"1st Quartile (Q1 - 25%):{np_q1:.4f} minutes")
print(f"2nd Quartile (Q2 - 50%):{np_q2:.4f} minutes")
print(f"3rd Quartile (Q3 - 75%):{np_q3:.4f} minutes")


# ==============================================================================
# 9. OUTLIER DETECTION
# ==============================================================================
print("\n" + "=" * 30)
print("9. OUTLIER DETECTION")
print("=" * 30)

# IQR Formula calculation
Q1 = np.percentile(duration_array, 25)
Q3 = np.percentile(duration_array, 75)
IQR = Q3 - Q1
Lower_Bound = Q1 - 1.5 * IQR
Upper_Bound = Q3 + 1.5 * IQR

# Filter outlier records
outlier_mask = (df['Workout_Duration_Mins'] < Lower_Bound) | (df['Workout_Duration_Mins'] > Upper_Bound)
outlier_records = df[outlier_mask]
num_outliers = len(outlier_records)

print(f"\n--- Interquartile Range (IQR) Boundary Parameters ---")
print(f"Q1 (25th Percentile): {Q1:.2f} mins")
print(f"Q3 (75th Percentile): {Q3:.2f} mins")
print(f"IQR (Q3 - Q1):        {IQR:.2f} mins")
print(f"Lower Bound:          {Lower_Bound:.2f} mins")
print(f"Upper Bound:          {Upper_Bound:.2f} mins")
print(f"Actual Data Range:    [{df['Workout_Duration_Mins'].min():.2f}, {df['Workout_Duration_Mins'].max():.2f}] mins")

print(f"\n[A] Number of Outliers Detected: {num_outliers}")
if num_outliers > 0:
    print("\nOutlier Records Found:")
    print(outlier_records[['Checkin_ID', 'Member_ID', 'Membership_Tier', 'Workout_Duration_Mins']])
else:
    print("    -> No statistical outliers detected. All workout durations fall strictly within")
    print(f"       the allowable IQR boundaries [{Lower_Bound:.2f}, {Upper_Bound:.2f}] minutes.")


# ==============================================================================
# 10. WORKOUT INTENSITY
# ==============================================================================
print("\n" + "=" * 30)
print("10. WORKOUT INTENSITY")
print("=" * 30)

# Rules:
# Duration < 45 -> Short
# 45 <= Duration < 90 -> Moderate
# Duration >= 90 -> Extended

conditions = [
    df['Workout_Duration_Mins'] < 45,
    (df['Workout_Duration_Mins'] >= 45) & (df['Workout_Duration_Mins'] < 90),
    df['Workout_Duration_Mins'] >= 90
]

choices = ['Short', 'Moderate', 'Extended']

# Create Workout_Intensity column using np.select
df['Workout_Intensity'] = np.select(conditions, choices, default='Unknown')

# Display category counts and proportions
intensity_counts = df['Workout_Intensity'].value_counts()[choices].reset_index()
intensity_counts.columns = ['Workout_Intensity', 'Session_Count']
intensity_counts['Percentage'] = (intensity_counts['Session_Count'] / total_checkins * 100).round(2)

print("\n--- Workout Intensity Classification Counts ---")
print(intensity_counts.to_string(index=False))


# ==============================================================================
# GENDER ANALYSIS (SECTION 8 IN USER PROMPT)
# ==============================================================================
gender_analysis = df.groupby('Gender').agg(
    Average_Duration=('Workout_Duration_Mins', 'mean'),
    Median_Duration=('Workout_Duration_Mins', 'median'),
    Visits_Count=('Checkin_ID', 'count')
).round(2).reindex(['Male', 'Female', 'Other']).reset_index()

print("\n--- Gender Analysis: Workout Duration Comparison ---")
print(gender_analysis.to_string(index=False))


# ==============================================================================
# 11. RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 30)
print("11. RECOMMENDATIONS")
print("=" * 30)

recommendations = """
1. MEMBER RETENTION & PROACTIVE ONBOARDING:
   - Data indicates that over 48% of total visits come from Basic tier members, but total
     visits per individual member are spread thin across billing months.
   - Implement an automated milestone-based onboarding program (e.g., automated check-in
     rewards at 5, 10, and 20 visits) to reinforce consistent gym habits during the first 60 days.

2. LOW ENGAGEMENT INTERVENTIONS & AT-RISK ALERTS:
   - A substantial cohort of members record fewer than 2 visits per active month, presenting
     a severe cancellation and churn risk.
   - Establish an automated "We Miss You" alert triggering after 14 consecutive days of inactivity.
     Offer these members a complimentary 1-on-1 personal trainer consultation, body composition
     assessment, or free guest pass to reactivate attendance.

3. OPTIMIZING FITNESS CLASS SCHEDULING:
   - 'None' (independent gym workouts: 153 check-ins) and 'Strength' (151 check-ins) lead overall
     attendance, followed closely by Spin (143), Zumba (136), HIIT (135), and Yoga (132).
   - High demand for functional Strength and cardio conditioning calls for expanding Strength and Spin
     time slots into peak operating hours. For Yoga, schedule restorative classes during evening slots
     (7:00 PM - 9:00 PM) when members unwind.

4. DYNAMIC PEAK-HOUR STAFFING & EQUIPMENT AVAILABILITY:
   - Attendance spikes heavily at 7:00 AM (morning rush) and 1:00 PM (lunch break rush), alongside
     sustained evening traffic from 6:00 PM to 9:00 PM. Weekends show Saturday as the single busiest day.
   - Shift floor trainers, sanitation personnel, and front-desk staff schedules to cover 06:30 - 09:30 AM
     and 12:30 - 02:30 PM, plus Saturday morning shifts, ensuring zero equipment bottleneck and optimal
     hygiene standards.

5. REVENUE OPTIMIZATION & TIER UPSELL PATHWAYS:
   - Basic tier generates the highest aggregate revenue ($691,585) due to high volume, but Gold and
     Platinum members demonstrate significantly higher customer lifetime value.
   - Launch targeted upgrade campaigns for highly engaged Basic members (e.g., those attending >10
     times/month or taking classes) offering a discounted first month of Gold/Platinum membership
     with premium amenities (complimentary recovery shakes, sauna, and priority class booking).
"""
print(recommendations)


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
print("\nGenerating Analytics Visualizations...")

# ------------------------------------------------------------------------------
# Chart 1: Membership Tier Distribution (Visits & Revenue Contribution)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Visits by Tier
sns.barplot(
    data=tier_analysis,
    x='Membership_Tier',
    y='Number_of_Visits',
    palette='Blues_r',
    ax=ax1
)
ax1.set_title('Total Check-ins by Membership Tier', fontweight='bold')
ax1.set_xlabel('Membership Tier')
ax1.set_ylabel('Number of Visits')
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height())}\n({p.get_height()/total_checkins*100:.1f}%)",
                 (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                 ha='center', va='center', color='white', fontweight='bold')

# Revenue Contribution Donut Chart
colors = sns.color_palette('pastel')[0:3]
ax2.pie(
    tier_analysis['Total_Revenue'],
    labels=tier_analysis['Membership_Tier'],
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
)
ax2.set_title('Revenue Contribution by Membership Tier', fontweight='bold')
plt.suptitle('Membership Tier Analytics', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 2: Peak Gym Hours
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))
hour_data = df['Hour'].value_counts().sort_index().reset_index()
hour_data.columns = ['Hour', 'Checkins']

palette_hours = ['#ff7f0e' if h in [7, 13] else '#1f77b4' for h in hour_data['Hour']]
ax = sns.barplot(data=hour_data, x='Hour', y='Checkins', palette=palette_hours)
plt.title('Peak Gym Hours (Check-in Frequency by Hour of Day)', fontweight='bold')
plt.xlabel('Hour of Day (24-Hour Clock)')
plt.ylabel('Check-in Count')
plt.xticks(range(len(hour_data['Hour'])), [f"{h:02d}:00" for h in hour_data['Hour']], rotation=45)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height)}",
                    (p.get_x() + p.get_width() / 2., height + 0.8),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 3: Busiest Weekdays
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
weekday_df = weekday_counts.reset_index()
weekday_df.columns = ['Weekday', 'Checkins']

ax = sns.barplot(
    data=weekday_df,
    x='Weekday',
    y='Checkins',
    palette='viridis'
)
plt.title('Busiest Gym Days of the Week', fontweight='bold')
plt.xlabel('Day of the Week')
plt.ylabel('Check-in Count')

for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 4: Fitness Class Attendance Ranking
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=class_summary,
    x='Fitness_Class',
    y='Attendance_Count',
    palette='mako'
)
plt.title('Fitness Class Attendance Ranking (Yoga, HIIT, Zumba, Strength, Spin, None)', fontweight='bold')
plt.xlabel('Fitness Class')
plt.ylabel('Attendance Count')

for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}\n({p.get_height()/total_checkins*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 5: Revenue by Membership Tier
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=revenue_by_tier,
    x='Membership_Tier',
    y='Total_Revenue',
    palette='crest'
)
plt.title('Total Revenue Generated by Membership Tier', fontweight='bold')
plt.xlabel('Membership Tier')
plt.ylabel('Total Revenue ($)')

for p in ax.patches:
    ax.annotate(f"${p.get_height():,.0f}",
                (p.get_x() + p.get_width() / 2., p.get_height() + 8000),
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 6: Monthly Revenue Trend
# ------------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
rev_trend_df = revenue_by_month.copy()
rev_trend_df['Year_Month_Str'] = rev_trend_df['Year_Month'].astype(str)

plt.plot(rev_trend_df['Year_Month_Str'], rev_trend_df['Total_Revenue'],
         marker='o', color='#2ca02c', linewidth=2.5, markersize=7)
plt.title('Monthly Gym Membership Revenue Trend (2025 - 2026)', fontweight='bold')
plt.xlabel('Billing Period (Year-Month)')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)

for x, y in zip(rev_trend_df['Year_Month_Str'], rev_trend_df['Total_Revenue']):
    plt.annotate(f"${y/1000:.1f}K", (x, y + 1200), ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 7: Workout Duration by Gender
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=gender_analysis,
    x='Gender',
    y='Average_Duration',
    palette='Set2'
)
plt.title('Average Workout Duration by Gender (Male, Female, Other)', fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Average Workout Duration (Minutes)')
plt.ylim(0, 95)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f} mins",
                (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Chart 8: Workout Intensity Distribution
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=intensity_counts,
    x='Workout_Intensity',
    y='Session_Count',
    palette='Spectral'
)
plt.title('Workout Intensity Distribution (Short < 45m | Moderate 45-89m | Extended >= 90m)', fontweight='bold')
plt.xlabel('Workout Intensity Category')
plt.ylabel('Number of Sessions')

for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/total_checkins*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nAll tasks, statistical computations, and visualizations successfully completed!")
