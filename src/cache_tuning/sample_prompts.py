QUICKSORT_PROMPT = """You are given the following code
```python
def sort(array):
    '''Sort the array by using quicksort.'''

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        return sort(less)+equal+sort(greater)  # Just use the + operator to join lists
    else:
        return array
```"""
