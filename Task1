# Example: Print Hello, World!
print("Hello, World!")


# Adding two numbers
a = 5
b = 10
result = a + b
result

# Code Cell 1: Import necessary libraries
import numpy as np

# Code Cell 2: Initialize the list of natural numbers
natural_numbers = np.arange(1, 11)

# Code Cell 3: Calculate the sum
sum_of_numbers = np.sum(natural_numbers)

# Code Cell 4: Display the result
print("The sum of the first ten natural numbers is:", sum_of_numbers)


# Code Cell 1: Import necessary libraries
import pandas as pd

# Code Cell 2: Create a dictionary with student data
data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Math': [85, 92, 78, 90, 88],
    'Science': [91, 89, 85, 94, 92],
    'English': [88, 93, 79, 87, 85]
}

# Code Cell 3: Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Code Cell 4: Calculate the average score for each student
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

# Code Cell 5: Display the DataFrame
print(df)


# Code Cell 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Code Cell 2: Generate data for the sine function
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Code Cell 3: Plot the graph
plt.plot(x, y, label='Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Plot of the Sine Function')
plt.legend()
plt.grid(True)
plt.show()


# Plot a simple graph using matplotlib
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Plotting
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.show()


# Scatter plot using Matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
x = np.random.rand(50)
y = 2 * x + 1 + 0.1 * np.random.randn(50)

# Create a scatter plot
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# Line plot and histogram using Seaborn
import seaborn as sns
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Line plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title('Line Plot')
plt.legend()

# Histogram
plt.subplot(1, 2, 2)
data = np.random.randn(1000)
sns.histplot(data, kde=True, color='skyblue')
plt.title('Histogram')
plt.show()


# Create a Pandas DataFrame
import pandas as pd

# Data
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}

# Create DataFrame
df = pd.DataFrame(data)
df


# Using NumPy for numerical operations
import numpy as np

# Create a NumPy array and perform operations
array = np.array([1, 2, 3, 4, 5])
sum_result = np.sum(array)
mean_result = np.mean(array)

sum_result, mean_result


# Code Cell 1: Import necessary libraries
import pandas as pd

# Code Cell 2: Create a DataFrame with sales data
data = {
    'Product': ['A', 'B', 'C', 'D'],
    'Sales': [100, 150, 200, 130],
    'Profit': [20, 30, 50, 25]
}
df = pd.DataFrame(data)

# Code Cell 3: Calculate the total sales and profit
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()

# Code Cell 4: Display the results
print(f'Total Sales: {total_sales}')
print(f'Total Profit: {total_profit}')


# Sample list
elements = ['apple', 'banana', 'apple', 'orange', 'banana', 'banana']

# Initialize an empty dictionary to store the frequency count
frequency = {}

# Count the frequency of each element
for item in elements:
    if item in frequency:
        frequency[item] += 1
    else:
        frequency[item] = 1

# Display the frequency count
print("Frequency of elements:", frequency)


# Lists of keys and values
keys = ['name', 'age', 'city']
values = [seetha', 25, 'India']

# Create a dictionary using the zip function
person_info = dict(zip(keys, values))

# Display the dictionary
print("Person Information:", person_info)


# Lists of keys and values
keys = ['name', 'age', 'city']
values = ['Alice', 25, 'New York']

# Create a dictionary using the zip function
person_info = dict(zip(keys, values))

# Display the dictionary
print("Person Information:", person_info)


# List of numbers
numbers = [1, 2, 3, 4, 5]

# Create a dictionary with numbers as keys and their squares as values
squares = {num: num**2 for num in numbers}

# Display the dictionary
print("Squares of numbers:", squares)


# Dictionary with student data
students = {
    'Alice': {'Math': 85, 'Science': 91, 'English': 88},
    'Bob': {'Math': 92, 'Science': 89, 'English': 93},
    'Charlie': {'Math': 78, 'Science': 85, 'English': 79}
}

# Display the nested dictionary
print("Student Scores:", students)


# Code Cell 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Code Cell 2: Create a sample DataFrame
data = {
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'D': np.random.randn(100)
}

df = pd.DataFrame(data)

# Code Cell 3: Calculate the correlation matrix
correlation_matrix = df.corr()

# Code Cell 4: Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()


# Code Cell 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Code Cell 2: Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Values': [10, 20, 15, 25, 20, 30]
}

df = pd.DataFrame(data)

# Code Cell 3: Create a pivot table
pivot_table = df.pivot('Category', 'Subcategory', 'Values')

# Code Cell 4: Create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_table, annot=True, cmap='viridis')
plt.title('Pivot Table Heatmap')
plt.show()


# Code Cell 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Code Cell 2: Generate random data
data = np.random.rand(10, 12)

# Code Cell 3: Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data, annot=True, cmap='plasma')
plt.title('Heatmap of Random Data')
plt.show()


