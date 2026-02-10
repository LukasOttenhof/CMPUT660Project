#include <stdio.h>

// Function 1: simple addition
int add(int a, int b) {
    return a + b;
}

// Function 2: compute factorial
int factorial(int n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

// Function 3: check if a number is prime
int is_prime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            return 0;
        }
    }
    return 1;
}

// Main function
int main() {
    int x = 5, y = 3;
    printf("Add: %d\n", add(x, y));
    printf("Factorial: %d\n", factorial(x));
    printf("Is %d prime? %d\n", x, is_prime(x));
    return 0;
}
