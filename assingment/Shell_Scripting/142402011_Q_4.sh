#!/bin/bash
# Simple Address Book Program

ADDRESS_BOOK="addresh_book.txt"

# Create file if not exists
touch "$ADDRESS_BOOK"

# Function: Search
search_entry() {
    read -p "Enter search term: " term
    grep -i "$term" "$ADDRESS_BOOK" || echo "No matching records found."
}

# Function: Add
add_entry() {
    read -p "Enter Name: " name
    read -p "Enter Surname: " surname
    read -p "Enter Email: " email
    read -p "Enter Phone: " phone

    echo "You entered: $name,$surname,$email,$phone"
    read -p "Save this record? (y/n): " confirm
    if [[ $confirm == [Yy] ]]; then
        echo "$name,$surname,$email,$phone" >> "$ADDRESS_BOOK"
        echo "Record saved."
    else
        echo "Cancelled."
    fi
}

# Function: Remove
remove_entry() {
    read -p "Enter search term for record to remove: " term
    matches=$(grep -i "$term" "$ADDRESS_BOOK")
    if [[ -z $matches ]]; then
        echo "No matching records."
        return
    fi
    echo "Matching records:"
    echo "$matches"

    read -p "Enter exact record to remove (copy & paste from above): " record
    read -p "Are you sure you want to delete this record? (y/n): " confirm
    if [[ $confirm == [Yy] ]]; then
        grep -vF "$record" "$ADDRESS_BOOK" > temp.txt && mv temp.txt "$ADDRESS_BOOK"
        echo "Record removed."
    else
        echo "Cancelled."
    fi
}

# Function: Display all
display_entries() {
    if [[ ! -s "$ADDRESS_BOOK" ]]; then
        echo "Address book is empty."
    else
        column -t -s "," "$ADDRESS_BOOK"
    fi
}

# Main menu
while true; do
    echo ""
    echo "====== Address Book ======"
    echo "1. Search"
    echo "2. Add"
    echo "3. Remove"
    echo "4. Display All"
    echo "5. Exit"
    read -p "Choose an option [1-5]: " choice

    case $choice in
        1) search_entry ;;
        2) add_entry ;;
        3) remove_entry ;;
        4) display_entries ;;
        5) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option. Try again." ;;
    esac
done
