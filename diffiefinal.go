package main

import (
	"bufio"
	"crypto/rand"
	"fmt"
	"math/big"
	"os"
	"strings"
)

const BLOCK_SIZE = 8 // Number of bytes per block

// stringToBlocks converts a string to blocks of big integers
func stringToBlocks(s string) []*big.Int {
	var blocks []*big.Int
	bytes := []byte(s)

	// Pad the bytes to be a multiple of BLOCK_SIZE
	if len(bytes)%BLOCK_SIZE != 0 {
		padding := BLOCK_SIZE - (len(bytes) % BLOCK_SIZE)
		for i := 0; i < padding; i++ {
			bytes = append(bytes, byte(padding))
		}
	}

	// Convert each block to a big integer
	for i := 0; i < len(bytes); i += BLOCK_SIZE {
		block := new(big.Int)
		for j := 0; j < BLOCK_SIZE; j++ {
			block.Lsh(block, 8)
			block.Or(block, big.NewInt(int64(bytes[i+j])))
		}
		blocks = append(blocks, block)
	}
	return blocks
}

// blocksToString converts blocks of big integers back to a string
func blocksToString(blocks []*big.Int) string {
	var bytes []byte

	for _, block := range blocks {
		blockBytes := make([]byte, BLOCK_SIZE)
		temp := new(big.Int).Set(block)

		// Extract bytes from the block
		for j := BLOCK_SIZE - 1; j >= 0; j-- {
			b := new(big.Int)
			b.And(temp, big.NewInt(255))
			blockBytes[j] = byte(b.Int64())
			temp.Rsh(temp, 8)
		}
		bytes = append(bytes, blockBytes...)
	}

	// Remove padding if present
	if len(bytes) > 0 {
		padding := int(bytes[len(bytes)-1])
		if padding > 0 && padding <= BLOCK_SIZE {
			bytes = bytes[:len(bytes)-padding]
		}
	}

	return string(bytes)
}

// isPrime checks if a number is prime using the Miller-Rabin primality test
func isPrime(n *big.Int) bool {
	if n.Cmp(big.NewInt(2)) < 0 {
		return false
	}
	// Run Miller-Rabin test with first 5 small primes as bases
	smallPrimes := []int64{2, 3, 5, 7, 11}
	for _, a := range smallPrimes {
		if !millerRabinTest(n, big.NewInt(a)) {
			return false
		}
	}
	return true
}

// millerRabinTest performs the Miller-Rabin primality test
func millerRabinTest(n, a *big.Int) bool {
	if n.Cmp(big.NewInt(2)) == 0 {
		return true
	}
	if n.Cmp(big.NewInt(2)) < 0 || n.Bit(0) == 0 {
		return false
	}

	// Write n-1 as 2^r * d
	d := new(big.Int).Sub(n, big.NewInt(1))
	r := 0
	for d.Bit(0) == 0 {
		r++
		d.Rsh(d, 1)
	}

	// Witness loop
	x := new(big.Int).Exp(a, d, n)
	if x.Cmp(big.NewInt(1)) == 0 || x.Cmp(new(big.Int).Sub(n, big.NewInt(1))) == 0 {
		return true
	}

	for i := 0; i < r-1; i++ {
		x.Mul(x, x)
		x.Mod(x, n)
		if x.Cmp(new(big.Int).Sub(n, big.NewInt(1))) == 0 {
			return true
		}
		if x.Cmp(big.NewInt(1)) == 0 {
			return false
		}
	}
	return false
}

// generatePrime generates a prime number of the specified bit length
func generatePrime(bits int) *big.Int {
	for {
		// Generate random number
		n, err := rand.Prime(rand.Reader, bits)
		if err != nil {
			continue
		}
		if isPrime(n) {
			return n
		}
	}
}

// Replace the generateSafePrime function with this optimized version
func generateSafePrime(bits int) *big.Int {
	fmt.Printf("Generating %d-bit safe prime...\n", bits)
	
	for attempts := 0; attempts < 1000; attempts++ {
		// Generate a random prime q of (bits-1) length
		q, err := rand.Prime(rand.Reader, bits-1)
		if err != nil {
			continue
		}
		
		// Calculate p = 2q + 1
		p := new(big.Int).Mul(q, big.NewInt(2))
		p.Add(p, big.NewInt(1))
		
		// Use Go's built-in ProbablyPrime which is much more reliable
		// ProbablyPrime(20) gives extremely high confidence
		if p.ProbablyPrime(20) {
			fmt.Printf("Safe prime found after %d attempts\n", attempts+1)
			return p
		}
		
		// Print progress for larger bit sizes
		if bits >= 1024 && attempts > 0 && attempts%10 == 0 {
			fmt.Printf("Attempt %d...\n", attempts)
		}
	}
	
	// Fallback: if safe prime generation fails, use regular prime
	fmt.Println("Safe prime generation taking too long, using regular prime...")
	p, err := rand.Prime(rand.Reader, bits)
	if err != nil {
		panic("Failed to generate prime")
	}
	return p
}

// Also replace the isPrime function with this more efficient version
func isPrime2(n *big.Int) bool {
	// Use Go's built-in ProbablyPrime which is much more efficient
	// and reliable than custom Miller-Rabin implementation
	return n.ProbablyPrime(20)
}

// Optimize the findPrimitiveRoot function for better performance
func findPrimitiveRoot(p *big.Int) *big.Int {
	// For safe primes p = 2q + 1, we can use a more efficient approach
	// Check if p is actually a safe prime first
	pMinus1 := new(big.Int).Sub(p, big.NewInt(1))
	q := new(big.Int).Div(pMinus1, big.NewInt(2))
	
	// If p = 2q + 1 where q is prime, then g is a primitive root if:
	// g^2 ≢ 1 (mod p) and g^q ≢ 1 (mod p)
	
	// Try small values first (they're more likely to be primitive roots)
	candidates := []*big.Int{
		big.NewInt(2), big.NewInt(3), big.NewInt(5), big.NewInt(7),
		big.NewInt(11), big.NewInt(13), big.NewInt(17), big.NewInt(19),
	}
	
	for _, g := range candidates {
		if g.Cmp(p) >= 0 {
			continue
		}
		
		// Check if g^2 ≢ 1 (mod p)
		g2 := new(big.Int).Exp(g, big.NewInt(2), p)
		if g2.Cmp(big.NewInt(1)) == 0 {
			continue
		}
		
		// Check if g^q ≢ 1 (mod p)
		gq := new(big.Int).Exp(g, q, p)
		if gq.Cmp(big.NewInt(1)) == 0 {
			continue
		}
		
		return new(big.Int).Set(g)
	}
	
	// If small candidates don't work, search systematically
	for g := big.NewInt(2); g.Cmp(p) < 0; g.Add(g, big.NewInt(1)) {
		// Check if g^2 ≢ 1 (mod p)
		g2 := new(big.Int).Exp(g, big.NewInt(2), p)
		if g2.Cmp(big.NewInt(1)) == 0 {
			continue
		}
		
		// Check if g^q ≢ 1 (mod p)
		gq := new(big.Int).Exp(g, q, p)
		if gq.Cmp(big.NewInt(1)) == 0 {
			continue
		}
		
		return new(big.Int).Set(g)
	}
	
	// This should never happen for a proper safe prime
	return big.NewInt(2)
}
// generateRandomInRange generates a random number in the range [2, max-1]
func generateRandomInRange(max *big.Int) *big.Int {
	// Generate random number in range [2, max-1]
	maxMinus2 := new(big.Int).Sub(max, big.NewInt(2))
	r, err := rand.Int(rand.Reader, maxMinus2)
	if err != nil {
		panic(err)
	}
	r.Add(r, big.NewInt(2))
	return r
}

// ===== RSA FUNCTIONS =====

// generateKeys generates public and private key components
func generateKeys(bits int) (p, q, n, phi, e, d *big.Int) {
	fmt.Printf("\nGenerating %d-bit prime numbers...\n", bits)

	// Generate p and q
	p = generatePrime(bits)
	for {
		q = generatePrime(bits)
		if p.Cmp(q) != 0 {
			break
		}
	}

	// Calculate n = p * q
	n = new(big.Int).Mul(p, q)

	// Calculate φ(n) = (p-1)(q-1)
	p1 := new(big.Int).Sub(p, big.NewInt(1))
	q1 := new(big.Int).Sub(q, big.NewInt(1))
	phi = new(big.Int).Mul(p1, q1)

	// Choose e = 65537 (common choice)
	e = big.NewInt(65537)

	// Calculate d (multiplicative inverse of e modulo phi)
	d = new(big.Int).ModInverse(e, phi)

	return
}

// RSA Attack Function
func performRSAAttack(encryptedMsg, n, e *big.Int, percentageIncrease float64) (*big.Int, error) {
	fmt.Println("\n=== RSA ATTACK SYSTEM ===")
	fmt.Printf("Original encrypted message: %v\n", encryptedMsg)
	fmt.Printf("Modulus (n): %v\n", n)
	fmt.Printf("Public exponent (e): %v\n", e)
	fmt.Printf("Percentage increase: %.2f%%\n", percentageIncrease)

	// Calculate the multiplier for the percentage increase
	multiplierNum := big.NewInt(int64(100 + percentageIncrease))
	multiplierDen := big.NewInt(100)

	fmt.Printf("Multiplier: %v/%v = %.4f\n", multiplierNum, multiplierDen, float64(multiplierNum.Int64())/float64(multiplierDen.Int64()))

	actualMultiplier := 1.0 + percentageIncrease/100.0
	
	// Scale up to work with integers while maintaining precision
	scaleFactor := int64(10000) // Use 4 decimal places precision
	scaledMultiplier := big.NewInt(int64(actualMultiplier * float64(scaleFactor)))
	scaleFactorBig := big.NewInt(scaleFactor)
	
	fmt.Printf("Scaled multiplier: %v/%v = %.6f\n", scaledMultiplier, scaleFactorBig, actualMultiplier)

	// Calculate (scaledMultiplier^e) mod n
	scaledMultiplierPowE := new(big.Int).Exp(scaledMultiplier, e, n)
	
	// Calculate (scaleFactor^e) mod n  
	scaleFactorPowE := new(big.Int).Exp(scaleFactorBig, e, n)
	
	// Calculate modular inverse of (scaleFactor^e) mod n
	scaleFactorPowEInv := new(big.Int).ModInverse(scaleFactorPowE, n)
	if scaleFactorPowEInv == nil {
		return nil, fmt.Errorf("cannot compute modular inverse - scale factor and n are not coprime")
	}
	
	// Calculate the effective multiplier: (scaledMultiplier^e) * (scaleFactor^e)^(-1) mod n
	effectiveMultiplier := new(big.Int).Mul(scaledMultiplierPowE, scaleFactorPowEInv)
	effectiveMultiplier.Mod(effectiveMultiplier, n)
	
	fmt.Printf("Effective multiplier for encryption: %v\n", effectiveMultiplier)

	// Calculate the new encrypted message: c' = c * effectiveMultiplier mod n
	newEncryptedMsg := new(big.Int).Mul(encryptedMsg, effectiveMultiplier)
	newEncryptedMsg.Mod(newEncryptedMsg, n)

	fmt.Printf("New encrypted message: %v\n", newEncryptedMsg)

	return newEncryptedMsg, nil
}

// Test the attack by decrypting and verifying
func testAttack(originalMsg, newEncryptedMsg, n, d *big.Int, expectedIncrease float64) {
	fmt.Println("\n=== ATTACK VERIFICATION ===")
	
	// Decrypt the new encrypted message
	decryptedMsg := new(big.Int).Exp(newEncryptedMsg, d, n)
	fmt.Printf("Decrypted manipulated message: %v\n", decryptedMsg)
	
	// Calculate the actual increase
	diff := new(big.Int).Sub(decryptedMsg, originalMsg)
	actualIncrease := float64(diff.Int64()) / float64(originalMsg.Int64()) * 100
	
	fmt.Printf("Original message: %v\n", originalMsg)
	fmt.Printf("Manipulated decrypted message: %v\n", decryptedMsg)
	fmt.Printf("Expected increase: %.2f%%\n", expectedIncrease)
	fmt.Printf("Actual increase: %.2f%%\n", actualIncrease)
	
	tolerance := 1.0 // 1% tolerance
	if actualIncrease >= expectedIncrease-tolerance && actualIncrease <= expectedIncrease+tolerance {
		fmt.Println("✓ Attack successful! Increase matches expected value within tolerance.")
	} else {
		fmt.Println("✗ Attack result differs from expected increase.")
	}
}

// ===== DIFFIE-HELLMAN FUNCTIONS =====

type DHParams struct {
	P *big.Int // Large prime
	G *big.Int // Generator (primitive root)
}

type DHKeyPair struct {
	PrivateKey *big.Int // Private key (random)
	PublicKey  *big.Int // Public key (g^private mod p)
}

// generateDHParams generates Diffie-Hellman parameters (p, g)
func generateDHParams(bits int) DHParams {
	fmt.Printf("Generating %d-bit Diffie-Hellman parameters...\n", bits)
	
	// Generate a safe prime p
	p := generateSafePrime(bits)
	
	// Find a primitive root g
	g := findPrimitiveRoot(p)
	
	fmt.Printf("Generated prime p: %v\n", p)
	fmt.Printf("Generated generator g: %v\n", g)
	
	return DHParams{P: p, G: g}
}

// generateDHKeyPair generates a Diffie-Hellman key pair
func generateDHKeyPair(params DHParams) DHKeyPair {
	// Generate private key: random number in range [2, p-2]
	privateKey := generateRandomInRange(params.P)
	
	// Generate public key: g^privateKey mod p
	publicKey := new(big.Int).Exp(params.G, privateKey, params.P)
	
	return DHKeyPair{
		PrivateKey: privateKey,
		PublicKey:  publicKey,
	}
}

// computeSharedSecret computes the shared secret using other party's public key
func computeSharedSecret(otherPublicKey *big.Int, myPrivateKey *big.Int, p *big.Int) *big.Int {
	// Shared secret = otherPublicKey^myPrivateKey mod p
	sharedSecret := new(big.Int).Exp(otherPublicKey, myPrivateKey, p)
	return sharedSecret
}

// demonstrateDiffieHellman demonstrates the Diffie-Hellman key exchange
func demonstrateDiffieHellman() {
	fmt.Println("\n=== DIFFIE-HELLMAN KEY EXCHANGE DEMONSTRATION ===")
	
	// Step 1: Generate parameters
	fmt.Println("\nStep 1: Generating public parameters")
	params := generateDHParams(512) // Use 512 bits for demo (use 2048+ in production)
	
	// Step 2: Alice generates her key pair
	fmt.Println("\nStep 2: Alice generates her key pair")
	aliceKeys := generateDHKeyPair(params)
	fmt.Printf("Alice's private key: %v\n", aliceKeys.PrivateKey)
	fmt.Printf("Alice's public key: %v\n", aliceKeys.PublicKey)
	
	// Step 3: Bob generates his key pair
	fmt.Println("\nStep 3: Bob generates his key pair")
	bobKeys := generateDHKeyPair(params)
	fmt.Printf("Bob's private key: %v\n", bobKeys.PrivateKey)
	fmt.Printf("Bob's public key: %v\n", bobKeys.PublicKey)
	
	// Step 4: Compute shared secrets
	fmt.Println("\nStep 4: Computing shared secrets")
	aliceSharedSecret := computeSharedSecret(bobKeys.PublicKey, aliceKeys.PrivateKey, params.P)
	bobSharedSecret := computeSharedSecret(aliceKeys.PublicKey, bobKeys.PrivateKey, params.P)
	
	fmt.Printf("Alice's computed shared secret: %v\n", aliceSharedSecret)
	fmt.Printf("Bob's computed shared secret: %v\n", bobSharedSecret)
	
	// Step 5: Verify
	fmt.Println("\nStep 5: Verification")
	if aliceSharedSecret.Cmp(bobSharedSecret) == 0 {
		fmt.Println("✓ Success! Both parties computed the same shared secret.")
	} else {
		fmt.Println("✗ Error! Shared secrets don't match.")
	}
}

// ===== ELGAMAL FUNCTIONS =====

type ElGamalParams struct {
	P *big.Int // Large prime
	G *big.Int // Generator
}

type ElGamalKeyPair struct {
	PublicKey  *big.Int // Public key (g^x mod p)
	PrivateKey *big.Int // Private key (x)
}

type ElGamalCiphertext struct {
	C1 *big.Int // g^k mod p
	C2 *big.Int // m * y^k mod p
}

// generateElGamalParams generates ElGamal parameters
func generateElGamalParams(bits int) ElGamalParams {
	fmt.Printf("Generating %d-bit ElGamal parameters...\n", bits)
	
	// Generate a safe prime p
	p := generateSafePrime(bits)
	
	// Find a primitive root g
	g := findPrimitiveRoot(p)
	
	return ElGamalParams{P: p, G: g}
}

// generateElGamalKeyPair generates an ElGamal key pair
func generateElGamalKeyPair(params ElGamalParams) ElGamalKeyPair {
	// Generate private key: random number in range [2, p-2]
	privateKey := generateRandomInRange(params.P)
	
	// Generate public key: g^privateKey mod p
	publicKey := new(big.Int).Exp(params.G, privateKey, params.P)
	
	return ElGamalKeyPair{
		PublicKey:  publicKey,
		PrivateKey: privateKey,
	}
}

// elGamalEncrypt encrypts a message using ElGamal
func elGamalEncrypt(message *big.Int, publicKey *big.Int, params ElGamalParams) ElGamalCiphertext {
	// Generate random k
	k := generateRandomInRange(params.P)
	
	// Compute c1 = g^k mod p
	c1 := new(big.Int).Exp(params.G, k, params.P)
	
	// Compute c2 = m * y^k mod p
	yk := new(big.Int).Exp(publicKey, k, params.P)
	c2 := new(big.Int).Mul(message, yk)
	c2.Mod(c2, params.P)
	
	return ElGamalCiphertext{C1: c1, C2: c2}
}

// elGamalDecrypt decrypts an ElGamal ciphertext
func elGamalDecrypt(ciphertext ElGamalCiphertext, privateKey *big.Int, params ElGamalParams) *big.Int {
	// Compute c1^x mod p
	c1x := new(big.Int).Exp(ciphertext.C1, privateKey, params.P)
	
	// Compute the modular inverse of c1^x mod p
	c1xInv := new(big.Int).ModInverse(c1x, params.P)
	
	// Compute message = c2 * (c1^x)^(-1) mod p
	message := new(big.Int).Mul(ciphertext.C2, c1xInv)
	message.Mod(message, params.P)
	
	return message
}

// demonstrateElGamal demonstrates ElGamal encryption/decryption
func demonstrateElGamal() {
	fmt.Println("\n=== ELGAMAL ENCRYPTION/DECRYPTION DEMONSTRATION ===")
	
	// Step 1: Generate parameters
	fmt.Println("\nStep 1: Generating ElGamal parameters")
	var bitLength int
	fmt.Print("Enter the bit length for the prime numbers: ")
	fmt.Scan(&bitLength)
	
	params := generateElGamalParams(bitLength)
	fmt.Printf("Prime p: %v\n", params.P)
	fmt.Printf("Generator g: %v\n", params.G)
	
	// Step 2: Generate key pair
	fmt.Println("\nStep 2: Generating key pair")
	keyPair := generateElGamalKeyPair(params)
	fmt.Printf("Private key (x): %v\n", keyPair.PrivateKey)
	fmt.Printf("Public key (y = g^x mod p): %v\n", keyPair.PublicKey)
	
	// Step 3: Get the message
	fmt.Println("\nStep 3: Enter the message")
	fmt.Println("Choose message type:")
	fmt.Println("1. Number")
	fmt.Println("2. Text")

	var choice int
	fmt.Print("Enter your choice (1 or 2): ")
	fmt.Scan(&choice)

	reader := bufio.NewReader(os.Stdin)

	if choice == 1 {
		// Number handling
		var msgNum int64
		fmt.Print("Enter a number: ")
		fmt.Scan(&msgNum)
		message := big.NewInt(msgNum)
		
		if message.Cmp(params.P) >= 0 {
			fmt.Println("Error: Message is too large for the current parameters.")
			fmt.Printf("Maximum value allowed: %v\n", params.P)
			return
		}
		
		fmt.Printf("\nOriginal Message (number): %v\n", message)
		
		// Step 4: Encrypt
		fmt.Println("\nStep 4: Encrypting message")
		fmt.Println("Formula: c1 = g^k mod p, c2 = m * y^k mod p")
		ciphertext := elGamalEncrypt(message, keyPair.PublicKey, params)
		fmt.Printf("Encrypted Message (c1, c2): (%v, %v)\n", ciphertext.C1, ciphertext.C2)
		
		// Step 5: Decrypt
		fmt.Println("\nStep 5: Decrypting message")
		fmt.Println("Formula: m = c2 * (c1^x)^(-1) mod p")
		decrypted := elGamalDecrypt(ciphertext, keyPair.PrivateKey, params)
		fmt.Printf("Decrypted Message: %v\n", decrypted)
		
		// Step 6: Verify
		fmt.Println("\nStep 6: Verification")
		if message.Cmp(decrypted) == 0 {
			fmt.Println("✓ Success! Original message recovered correctly.")
			fmt.Printf("Original (%v) = Decrypted (%v)\n", message, decrypted)
		} else {
			fmt.Println("✗ Error! Decryption failed.")
			fmt.Printf("Original (%v) ≠ Decrypted (%v)\n", message, decrypted)
		}

	} else {
		// Text handling
		fmt.Print("Enter text: ")
		reader.ReadString('\n')
		messageStr, _ := reader.ReadString('\n')
		messageStr = strings.TrimSpace(messageStr)

		fmt.Printf("\nOriginal Message: %v\n", messageStr)
		fmt.Printf("Block size: %d bytes\n", BLOCK_SIZE)

		// Convert message to blocks
		blocks := stringToBlocks(messageStr)
		fmt.Printf("Number of blocks: %d\n", len(blocks))

		// Encrypt each block
		fmt.Println("\nStep 4: Encrypting message blocks")
		var encryptedBlocks []ElGamalCiphertext
		for i, block := range blocks {
			fmt.Printf("\nBlock %d:\n", i+1)
			fmt.Printf("Original block value: %v\n", block)

			if block.Cmp(params.P) >= 0 {
				fmt.Printf("Error: Block %d is too large for the current key size.\n", i+1)
				return
			}

			encrypted := elGamalEncrypt(block, keyPair.PublicKey, params)
			fmt.Printf("Encrypted block (c1, c2): (%v, %v)\n", encrypted.C1, encrypted.C2)
			encryptedBlocks = append(encryptedBlocks, encrypted)
		}

		// Decrypt each block
		fmt.Println("\nStep 5: Decrypting message blocks")
		var decryptedBlocks []*big.Int
		for i, block := range encryptedBlocks {
			fmt.Printf("\nBlock %d:\n", i+1)
			decrypted := elGamalDecrypt(block, keyPair.PrivateKey, params)
			fmt.Printf("Decrypted block value: %v\n", decrypted)
			decryptedBlocks = append(decryptedBlocks, decrypted)
		}

		// Convert blocks back to string
		decryptedStr := blocksToString(decryptedBlocks)

		// Step 6: Verification
		fmt.Println("\nStep 6: Verification")
		if messageStr == decryptedStr {
			fmt.Println("✓ Success! Original message recovered correctly.")
			fmt.Printf("Original text: %v\n", messageStr)
			fmt.Printf("Decrypted text: %v\n", decryptedStr)
		} else {
			fmt.Println("✗ Error! Decryption failed.")
			fmt.Printf("Original text: %v\n", messageStr)
			fmt.Printf("Decrypted text: %v\n", decryptedStr)
		}
	}
}
func performRSAOperation() {
	// Step 1: Generate key components
	var bitLength int
	fmt.Println("\nStep 1: Generating RSA key components")
	fmt.Print("Enter the bit length for the prime numbers: ")
	fmt.Scan(&bitLength)
	p, q, n, phi, e, d := generateKeys(bitLength)

	fmt.Printf("p = %v (prime 1)\n", p)
	fmt.Printf("q = %v (prime 2)\n", q)
	fmt.Printf("n = p * q = %v (modulus)\n", n)
	fmt.Printf("φ(n) = (p-1)(q-1) = %v (Euler's totient)\n", phi)
	fmt.Printf("e = %v (public exponent)\n", e)
	fmt.Printf("d = %v (private exponent)\n", d)

	// Step 2: Get the message
	fmt.Println("\nStep 2: Enter the message")
	fmt.Println("Choose message type:")
	fmt.Println("1. Number")
	fmt.Println("2. Text")

	var choice int
	fmt.Print("Enter your choice (1 or 2): ")
	fmt.Scan(&choice)

	reader := bufio.NewReader(os.Stdin)

	if choice == 1 {
		var num int64
		fmt.Print("Enter a number: ")
		fmt.Scan(&num)
		messageInt := big.NewInt(num)

		if messageInt.Cmp(n) >= 0 {
			fmt.Println("\nError: Number is too large for the current key size.")
			fmt.Printf("Maximum value allowed: %v\n", n)
			return
		}

		fmt.Printf("\nOriginal Message (number): %v\n", messageInt)

		// Encryption
		fmt.Println("\nStep 3: Encrypting message")
		fmt.Println("Formula: ciphertext = message^e mod n")
		ciphertext := new(big.Int).Exp(messageInt, e, n)
		fmt.Printf("Encrypted Message: %v\n", ciphertext)

		// Decryption
		fmt.Println("\nStep 4: Decrypting message")
		fmt.Println("Formula: decrypted = ciphertext^d mod n")
		decrypted := new(big.Int).Exp(ciphertext, d, n)
		fmt.Printf("Decrypted Message: %v\n", decrypted)

		// Verification
		fmt.Println("\nStep 5: Verification")
		if messageInt.Cmp(decrypted) == 0 {
			fmt.Println("✓ Success! Original message recovered correctly.")
			fmt.Printf("Original (%v) = Decrypted (%v)\n", messageInt, decrypted)
		} else {
			fmt.Println("✗ Error! Decryption failed.")
			fmt.Printf("Original (%v) ≠ Decrypted (%v)\n", messageInt, decrypted)
		}

		// Optional demo of attack system
		fmt.Print("\nWould you like to demonstrate the attack system? (y/n): ")
		var demo string
		fmt.Scan(&demo)
		
		if demo == "y" || demo == "Y" {
			fmt.Println("\n=== ATTACK DEMONSTRATION ===")
			fmt.Print("Enter percentage increase for attack demo (e.g., 50 for 50%): ")
			var attackPercent float64
			fmt.Scan(&attackPercent)
			
			attackResult, err := performRSAAttack(ciphertext, n, e, attackPercent)
			if err != nil {
				fmt.Printf("Attack failed: %v\n", err)
			} else {
				testAttack(messageInt, attackResult, n, d, attackPercent)
			}
		}

	} else {
		// Text handling code
		fmt.Print("Enter text: ")
		reader.ReadString('\n')
		messageStr, _ := reader.ReadString('\n')
		messageStr = strings.TrimSpace(messageStr)

		fmt.Printf("\nOriginal Message: %v\n", messageStr)
		fmt.Printf("Block size: %d bytes\n", BLOCK_SIZE)

		// Convert message to blocks
		blocks := stringToBlocks(messageStr)
		fmt.Printf("Number of blocks: %d\n", len(blocks))

		// Encrypt each block
		fmt.Println("\nStep 3: Encrypting message blocks")
		var encryptedBlocks []*big.Int
		for i, block := range blocks {
			fmt.Printf("\nBlock %d:\n", i+1)
			fmt.Printf("Original block value: %v\n", block)

			if block.Cmp(n) >= 0 {
				fmt.Printf("Error: Block %d is too large for the current key size.\n", i+1)
				return
			}

			encrypted := new(big.Int).Exp(block, e, n)
			fmt.Printf("Encrypted block value: %v\n", encrypted)
			encryptedBlocks = append(encryptedBlocks, encrypted)
		}

		// Decrypt each block
		fmt.Println("\nStep 4: Decrypting message blocks")
		var decryptedBlocks []*big.Int
		for i, block := range encryptedBlocks {
			fmt.Printf("\nBlock %d:\n", i+1)
			decrypted := new(big.Int).Exp(block, d, n)
			fmt.Printf("Decrypted block value: %v\n", decrypted)
			decryptedBlocks = append(decryptedBlocks, decrypted)
		}

		// Convert blocks back to string
		decryptedStr := blocksToString(decryptedBlocks)

		// Verification
		fmt.Println("\nStep 5: Verification")
		if messageStr == decryptedStr {
			fmt.Println("✓ Success! Original message recovered correctly.")
			fmt.Printf("Original text: %v\n", messageStr)
			fmt.Printf("Decrypted text: %v\n", decryptedStr)
		} else {
			fmt.Println("✗ Error! Decryption failed.")
			fmt.Printf("Original text: %v\n", messageStr)
			fmt.Printf("Decrypted text: %v\n", decryptedStr)
		}
	}
}

func performRSAAttackOperation() {
	fmt.Println("\n=== RSA ATTACK SYSTEM ===")
	
	var encryptedMsgStr, nStr, eStr string
	var percentageIncrease float64
	
	fmt.Print("Enter the encrypted message: ")
	fmt.Scan(&encryptedMsgStr)
	
	fmt.Print("Enter the modulus (n): ")
	fmt.Scan(&nStr)
	
	fmt.Print("Enter the public exponent (e): ")
	fmt.Scan(&eStr)
	
	fmt.Print("Enter the percentage increase (e.g., 50 for 50%): ")
	fmt.Scan(&percentageIncrease)
	
	// Parse the input values
	encryptedMsg, ok1 := new(big.Int).SetString(encryptedMsgStr, 10)
	n, ok2 := new(big.Int).SetString(nStr, 10)
	e, ok3 := new(big.Int).SetString(eStr, 10)
	
	if !ok1 || !ok2 || !ok3 {
		fmt.Println("Error: Invalid input format for big integers")
		return
	}
	
	// Perform the attack
	newEncryptedMsg, err := performRSAAttack(encryptedMsg, n, e, percentageIncrease)
	if err != nil {
		fmt.Printf("Attack failed: %v\n", err)
		return
	}
	
	fmt.Printf("\n=== ATTACK RESULT ===\n")
	fmt.Printf("Original encrypted message: %v\n", encryptedMsg)
	fmt.Printf("Manipulated encrypted message: %v\n", newEncryptedMsg)
	fmt.Printf("When decrypted, this should give a value %.2f%% higher than the original\n", percentageIncrease)
	
	// Optional: If user provides private key for testing
	fmt.Print("\nDo you want to test the attack? Enter the private exponent (d) or press Enter to skip: ")
	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n') // consume the newline
	dStr, _ := reader.ReadString('\n')
	dStr = strings.TrimSpace(dStr)
	
	if dStr != "" {
		d, ok := new(big.Int).SetString(dStr, 10)
		if ok {
			// We need the original message to test
			originalMsg := new(big.Int).Exp(encryptedMsg, d, n)
			testAttack(originalMsg, newEncryptedMsg, n, d, percentageIncrease)
		}
	}
}
// ===== MAIN FUNCTION =====

func main() {
	fmt.Println("Advanced Cryptography System")
	fmt.Println("============================")
	fmt.Println("1. RSA Encryption/Decryption")
	fmt.Println("2. RSA Attack System")
	fmt.Println("3. Diffie-Hellman Key Exchange")
	fmt.Println("4. ElGamal Encryption/Decryption")
	
	var operation int
	fmt.Print("Enter your choice (1-4): ")
	fmt.Scan(&operation)

	switch operation {
	case 1:
		// RSA Normal Operation
		performRSAOperation()
	case 2:
		// RSA Attack System
		performRSAAttackOperation()
	case 3:
		// Diffie-Hellman
		demonstrateDiffieHellman()
	case 4:
		// ElGamal
		demonstrateElGamal()
	default:
		fmt.Println("Invalid choice!")
	}
}

