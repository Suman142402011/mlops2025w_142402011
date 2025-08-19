#!/bin/bash

clear
echo "--- Sum and Factorial of n numbers ---"
echo -n "Enter the number: "
read digit

t=1
total_sum=0
total_pro=1

while [ $t -le $digit ]
do
  total_sum=`expr $total_sum + $t`
  total_pro=`expr $total_pro \* $t`
  t=`expr $t + 1`
done

echo "SUM OF $digit : $total_sum"
echo "Factorial of $digit : $total_pro"

