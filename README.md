# Joshua Whitton Python-Portfolio
This is the portfolio final project for Dr. Vandenbrinks BISC 450 C at Louisiana Tech University. This Project was completed by Joshua Whitton, a Senior at Louisiana Tech University who is on the Pre-Medical Route. This class was taken during the first Summer Session at Louisiana Tech University during the 2025 school year

## Using Jupyter Notebooks (1 and 2)
### In this section we learned the basics of using Jupyter Notebooks

```python
%matplotlib inline
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv ('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```


```python
df.tail()
```


```python
df.columns = ['year', 'rank', 'company','revenue', 'profit']
```


```python
df.head()
```


```python
len(df)
```


```python
df.dtypes
```


```python
non_numeric_profits = df.profit.str.contains ('[^0-9.-]')
df.loc[non_numeric_profits].head()
```


```python
set(df.profit[non_numeric_profits])
```


```python
len(df.profit[non_numeric_profits])
```


```python
bin_sizes, _, _ =plt.hist(df.year[non_numeric_profits], bins=range(1955,2006))
```


```python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```


```python
df.dtypes
```


```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x=0, y=0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot (x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue(millions)')
```


```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2)= plt.subplots(ncols=2)
title = 'Increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14,4)
fig.tight_layout()
```
### Python Fundamentals
In this section we learned the basic fundamentals of Python 
```python
# Any python interpreter can be used as a calculator:
3 + 5 * 4
```




    23




```python
# Lets save a value to a variable
weight_kg = 60
```


```python
print(weight_kg)
```

    60



```python
# Weight0 = valud
# 0weight = invalid
# weight and Weight are different
```


```python
# Types of data 
# There are three common types of data
# Interger numbers
# floating point numbers 
# Strings

```


```python
# floating point number
weight_kg = 60.3 
```


```python
# String comprised of letters 
Patient_name = "Jon Smith"
```


```python
# String comprised of numbers
patient_id = '001'
```


```python
# Use variables in python
weight_lb = 2.2 * weight_kg


print(weight_lb)
```

    132.66



```python
# lets add a prefix to our patient id 

patient_id = 'inflam_'+ patient_id

print(patient_id)
```

    inflam_001



```python
# lets combine print statements 
print(patient_id, 'weight in kg', weight_kg)
```

    inflam_001 weight in kg 60.3



```python
# we can call a function inside another function 

print(type(60.3))

print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
# we can also do calculations inside different functions 

print('weight ibs', 2.2 * weight_kg)
```

    weight ibs 132.66



```python
print(weight_kg)
```

    60.3



```python
weight_kg= 65.0 
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0

### Analyzing Data (1,2, and 3)
In this section we learned how to analyze inflammation data from various patients
```python
import numpy
```


```python
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.shape)
```

    (60, 40)



```python
print('first value in data:', data [0,0])
```

    first value in data: 0.0



```python
print('middle value in data:', data [29,19])
```

    middle value in data: 16.0



```python
print(data[0:4,0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data [:3, 36:]
```


```python
print('small is')
```

    small is



```python
print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# Lets use a numpy function
print(numpy.mean(data))
```

    6.14875



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)

```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
print('maximum inflammation', maxval)
print('minimum inflammation', minval)
print('standard deviation', stdval)
```

    maximum inflammation 20.0
    minimum inflammation 0.0
    standard deviation 4.613833197118566



```python
# Sometimes we want to look at variation in statistical values, such as maximum inflammation per patient,
#or average from day one

patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)

print('maxium inflammation for patient 0:', numpy.amax(patient_0))
```

    maxium inflammation for patient 0: 18.0



```python
print('maximum inflammation for patient 2:', numpy.amax(data[2, :]))
```

    maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data,axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(numpy.mean(data,axis = 0).shape)
```

    (40,)



```python
print(numpy.mean(data, axis = 1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



```python
import numpy
data = numpy.loadtxt(fname= 'inflammation-01.csv', delimiter = ',')
```


```python
# Heat map of patient inflammation over time
import matplotlib.pyplot
image = matplotlib.pyplot.imshow(data)
matplotlib.pyplot.show()
```


    <Figure size 640x480 with 1 Axes>



```python
# Average inflammation over time
ave_inflammation = numpy.mean(data, axis = 0)
ave_plot = matplotlib.pyplot.plot(ave_inflammation)
matplotlib.pyplot.show()
```

```python
max_plot = matplotlib.pyplot.plot(numpy.amax(data, axis =0))
matplotlib.pyplot.show()
```

```python
min_plot = matplotlib.pyplot.plot(numpy.amin(data, axis=0))
matplotlib.pyplot.show()
```

```python
fig = matplotlib.pyplot.figure(figsize =(10.0, 3.0))

axes1 = fig.add_subplot(1, 3, 1)
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel('average')
axes1.plot(numpy.mean(data, axis= 0))

axes2.set_ylabel('max')
axes2.plot(numpy.amax(data, axis = 0))

axes3.set_ylabel('min')
axes3.plot(numpy.amin(data, axis = 0))

fig.tight_layout()

matplotlib.pyplot.savefig('inflammation.png')
matplotlib.pyplot.show()


```

### Storing Values in Lists 
In this section we learned how values can be stored in lists
```python
odds = [1, 3, 5, 7]
print('odds are:', odds)
```

    odds are: [1, 3, 5, 7]



```python
print('first element:', odds[0])
print('last element:', odds[3])
print('"-1" element', odds[-1])
```

    first element: 1
    last element: 7
    "-1" element 7



```python
names = ['Curie', 'Darwing', 'Turing'] #Typo in Darwin's name
print('names is orginally:', names)
names[1]= 'Darwin' # Correct the name
print('final value of names:', names)
```

    names is orginally: ['Curie', 'Darwing', 'Turing']
    final value of names: ['Curie', 'Darwin', 'Turing']



```python
name = 'Darwin'
#name[0] = 'd'
```


```python
odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element: [3, 5, 7, 11]
    removed_element: 1



```python
odds.reverse()
print('odds after reversing:', odds)
```

    odds after reversing: [11, 7, 5, 3]



```python
odds = [3, 5, 7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
odds = [3, 5, 7]
primes = list(odds)
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]



```python
binomial_name = "Drosophilia melanogaster"
group = binomial_name[0:11]
print('group:', group)
species = binomial_name[11:24]
print('species:', species)
chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes [2:5]
print('autosomes', autosomes)
last = chromosomes[-1]
print('last:', last)
```

    group: Drosophilia
    species:  melanogaster
    autosomes ['2', '3', '4']
    last: 4



```python
date = "Monday 4 January 2023"
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning index', day)
```

    Using 0 to begin range: Monday
    Omitting beginning index Monday



```python
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position:', sond)

sond = months[8:len(months)]
print('Using len() to get last entry:', sond)
sond = months[8:]
print('Omitting ending index:', sond)
```

    With known last position: ['sep', 'oct', 'nov', 'dec']
    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
    Omitting ending index: ['sep', 'oct', 'nov', 'dec']

### Using Loops
In this section we learned the basics of using loops to increase coding efficiency 
```python
odds = [1, 3, 5, 7]
```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5
    7



```python
odds = [1, 3, 5,]
print(odds[0])
print(odds[1])
print(odds[2])


```

    1
    3
    5



```python
odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
for num in odds:
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19



```python
length = 0 
names = ['Curie', 'Darwin', 'Turing']
for value in names:
    length = length + 1 
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.



```python
name = 'Rosalind'
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is.', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is. Turing



```python
print(len([0,1,2,3]))
```

    4



```python
name = ['Curie', 'Darwin', 'Turing']
print(len(name))
```

    3
### Using Multiple Files
In this section we learned how to use multiple files using Python Commands
```python
import glob
```


```python
print(glob.glob('inflammation*.csv'))
```

    ['inflammation-10.csv', 'inflammation-09.csv', 'inflammation-11.csv', 'inflammation-06.csv', 'inflammation-05.csv', 'inflammation-08.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-04.csv', 'inflammation-03.csv', 'inflammation-02.csv', 'inflammation-12.csv']



```python
import glob
import numpy
import matplotlib.pyplot

filenames = sorted(glob.glob('inflammation*.csv'))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname=filename, delimiter=',')
    
    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))  # Added missing =
    
    axes1 = fig.add_subplot(1,3,1)  # Fixed: add_subplot (not add_sublot)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis=0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis=0))  # Fixed: amin (not amix), removed space
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```
### Making Choices 
How we can use Python to make choices 
```python
num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53
print('before conditional...')
if num > 100:
    print('num, is greater than 100')
print('...after conditional')
```

    before conditional...
    ...after conditional



```python
num = 14 
if num > 0: 
    print(num, 'is positive')
elif num ==0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    14 is positive



```python
if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else:
    print('at least one part is false')
```

    at least one part is false



```python
if (1 > 0) or (-1 >= 0):
    print('at least one part are true')

```

    at least one part are true



```python
if (1 > 0) or (-1 >= 0):
    print('at least on part are true')
else:
    print('both of these are false')
```

    at least on part are true
```python
import numpy
```


```python
numpy.loadtxt(fname='inflammation-01.csv', delimiter=',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
max_inflammation_0 = numpy.amax(data, axis=0)[0]
```


```python
max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 ==0 and max_inflammation_20 ==20:
    print('Saspictious looking Maxina!')

```

    Saspictious looking Maxina!



```python
max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 ==0 and max_inflammation_20 ==20:
    print('Saspictious looking Maxina!')
    
elif numpy.sum(numpy.amin(data, axis=0))==0:
    print(('Minima add up to Zero!'))
else:
    print('Seems Ok!')
```

    Saspictious looking Maxina!



```python
data = numpy.loadtxt(fname = 'inflammation-03.csv,delimiter= ',')
max_inflammation_0 = numpy.amax(data, axis = 0)[0]
max_inflammation_20 = numpy.amax(data, axis =0)[20]
max_inflammation_0 == 0 and max_inflammation_20 ==20:
                     print('Suspicious looking Maxima!')
elif numpy.sum(numpy.amin(data, axis=0)) ==0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems ok!')
```


      File "<ipython-input-12-c4c337b1ece3>", line 1
        data = numpy.loadtxt(fname = 'inflammation-03.csv,delimiter= ',')
                                                                         ^
    SyntaxError: EOL while scanning string literal




```python
data = numpy.loadtxt(fname='inflammation-03.csv', delimiter=',')  # Fixed quotes and delimiter
max_inflammation_0 = numpy.amax(data, axis=0)[0]
max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:  # Added 'if'
    print('Suspicious looking Maxima!')
elif numpy.sum(numpy.amin(data, axis=0)) == 0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems ok!')
```

    Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!

### Defensive Programming
In this section we learned the concept of defensive programming adnd the techniques behind its usage
```python
numbers = [1.5, 2.3, 0.7, 0.001, 4,4]
total = 0.0 
for num in numbers: 
    assert num > 0.0,'Data should only contain positive values'
    total += num
print('total is:', total)

```

    total is: 12.501000000000001



```python
def normalize_rectangle(rect):
    '''Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    Input should be of the format (x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and upper right corners of the rectangle respectively.'''
    
    assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid x coordinates'
    assert y0 < y1, 'Invalid y coordinates'
    
    dx = x1 - x0 
    dy = y1 - y0 
    if dx > dy:
        scaled = dy / dx
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dy / dx
        upper_x, upper_y = scaled, 1.0
        
    assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
    
    return (0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle( (0.0, 0.0, 1.0, 5.0)))
```

    (0, 0, 0.2, 1.0)
### Transcribing DNA into RNA 
in this section, we learned how to transcribe a sample of DNA into RNA 
```python
#prompt the user tp enter the input fasta file name

input_file_name = input("Enter the name of the input fasta file: ")
```


```python
#open the input fasta file and read the DNA sequence 
with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()

```


```python
# Transcribe the DNA to RNA
rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
# Prompt the user to enter the output file name
output_file_name = input("Enter the name of the output file: ")
```


```python
# Save the RNA sequence to a text file
with open(output_file_name,"w") as output_file:
    output_file.write(rna_sequence)
    print(f"The RNA sequence has been saved to {output_file_name}")
```


```python
print(rna_sequence)
```
### Translating RNA into Protein
In this sample, we transcribed the RNA found in the Transcription section of and converted it into its sequence of Amino Acids
```python
# Prompt the user to enter the input RNA file name
input_file_name = input("Enter the name of the input RNA file: ")
```

    Enter the name of the input RNA file:  SUMO_RNA.txt



```python
# Open the input RNA file and read the RNA sequence 

with open (input_file_name, "r") as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define Codon table

codon_table = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}

```


```python
# Translate RNA to Protein

protein_sequence = " "
for i in range (0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon)==3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


```python
# Prompt the user to enter the output file name

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  SUMO_Protein_txt



```python
# Save the protein sequence to a text file

with open(output_file_name, "w") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_file_name}")
```

    The protein sequence has been saved to SUMO_Protein_txt



```python
print(protein_sequence)

```

     MSDEKKGGETEHINLKVLGQDNAVVQFKIKKHTPLRKLMNAYCDRAGLSMQVVRFRFDGQPINENDTPTSLEMEEGDTIEVYQQQTGGAP



```



