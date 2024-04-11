

from numpy import float32, array

data = [
    (array([0.15725508], dtype=float32), ['Who Framed Roger Rabbit? (1988)']),
    (array([0.23456789], dtype=float32), ['Another Movie (1990)']),
    (array([0.12345678], dtype=float32), ['Yet Another Movie (2000)'])
]

sorted_data = sorted(data, key=lambda x: x[0][0], reverse=True)
print(sorted_data)
