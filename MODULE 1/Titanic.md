## Summary of Notebook Code and Their Purposes

1. **Import libraries**

   ```python
   import pandas as pd
   import numpy as np
   ```

   _Import pandas and numpy for data manipulation and analysis._

2. **Load the dataset**

   ```python
   df = pd.read_csv('Module.csv')
   df
   ```

   _Read the CSV file into a DataFrame and display it._

3. **DataFrame info**

   ```python
   df.info()
   ```

   _Show summary information about the DataFrame, including column types and non-null counts._

4. **Descriptive statistics**

   ```python
   df.describe()
   ```

   _Display basic statistics (mean, std, min, max, etc.) for numeric columns._

5. **Check for missing values**

   ```python
   df.isnull()
   df.isnull().sum()
   ```

   _Identify missing values and count them per column._

6. **Filter passengers older than 30**

   ```python
   df[df['age'] > 30]
   df[df['age'] > 30].head()  # First 5 passengers older than 30
   df[df['age'] > 30].tail()  # Last 5 passengers older than 30
   ```

   _Select passengers with age greater than 30 and view subsets._

7. **Filter male passengers**

   ```python
   df[df['sex'].isin(['male'])]  # Fetching male passengers
   df[~df['sex'].isin(['male'])] # Removing male passengers
   ```

   _Select or exclude male passengers._

8. **Embarked column analysis**

   ```python
   df['embarked'].value_counts()  # Count of passengers from each port
   df['embarked'].unique()        # Unique values in embarked column
   df['embarked'].nunique()       # Number of unique values in embarked column
   ```

   _Analyze the 'embarked' column for value counts and uniqueness._

9. **Create age groups**

   ```python
   df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 60, 80], labels=['child', 'teen', 'adult', 'senior'])
   df
   ```

   _Categorize passengers into age groups._

10. **High fare flag**

    ```python
    df['high_fare'] = df['fare'] > 100
    df.head(20)
    ```

    _Flag passengers who paid a fare greater than 100._

11. **Mean fare by sex**

    ```python
    mean_fare_by_sex = df.groupby('sex')['fare'].mean()
    mean_fare_by_sex
    ```

    _Calculate average fare for each sex._

12. **Mean age by sex and class**

    ```python
    mean_age_sex_class = df.groupby(['sex', 'class'])['age'].mean()
    mean_age_sex_class
    ```

    _Calculate average age grouped by sex and class._

13. **Aggregate statistics by class**

    ```python
    agg_stat = df.groupby('class').agg({'fare': ['mean', 'max', 'min'], 'age': 'median'})
    agg_stat
    ```

    _Get mean, max, min fare and median age for each class._

14. **Mean fare by class (transform)**

    ```python
    mean_fare_by_class = df.groupby('class')['fare'].transform('mean')
    mean_fare_by_class
    ```

    _Assign mean fare for each class to all rows._

15. **Mean age**

    ```python
    mean_age = df['age'].mean()
    print("Mean age:", mean_age)
    ```

    _Calculate and print the mean age._

16. **Mode of survived**

    ```python
    mode_survived = df['survived'].mode()
    print("Mode of survived:", mode_survived)
    ```

    _Find and print the most common value in the 'survived' column._

17. **Correlation analysis**

    ```python
    corr_matrix = df.corr(numeric_only=True)
    selected_corr = corr_matrix.loc[['age', 'fare', 'survived'], ['age', 'fare', 'survived']]
    print(selected_corr)
    ```

    _Calculate and print the correlation between age, fare, and survived._

18. **Correlation matrix for all numeric columns**
    ```python
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)
    ```
    _Display the correlation coefficients between all numeric columns._
