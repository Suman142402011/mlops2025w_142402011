
#1. Write a shell script to find the roots of the quadratic equation - 2 Marks


#!/bin/bash
# Quadratic equation: ax^2 + bx + c = 0

echo "enter a:"
read a
echo "enter b:"
read b
echo "enter c:"
read c

#discriminant
d=$(echo "$b*$b - 4*$a*$c" | bc -l)  

#if discriminant<0 then root imaginary
if (( $(echo "$d < 0" | bc -l) )); then
    echo "Roots are imaginary"

#If discriminant =0 then equal and real root 
elif (( $(echo "$d == 0" | bc -l) )); then
    root=$(echo "scale=2; -$b / (2*$a)" | bc -l)
    echo "Roots are real and equal: $root"

#and if discriminat >0 then all roots are distint and real
else
    sqrt_d=$(echo "scale=4; sqrt($d)" | bc -l)
    root1=$(echo "scale=4; (-$b + $sqrt_d) / (2*$a)" | bc -l)
    root2=$(echo "scale=4; (-$b - $sqrt_d) / (2*$a)" | bc -l)
    echo "Roots are real and different:"
    echo "Root 1 = $root1"
    echo "Root 2 = $root2"
fi
