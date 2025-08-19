clear
echo "--- Sum of n numbers ---"
echo -n "Enter the number: "
read digit
t=1
total=0

while [ $t -le $digit ]
do
  total=`expr $total + $t`
  t=`expr $t + 1`
done

echo "SUM OF $digit : $total"

