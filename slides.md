% Haskell: Introduction to Weird Functional Languages
% Alex Burkhart


# Part 1: Functional Programming

# The Functional Disipline

 - dis·ci·pline: (noun) activity, exercise, or a regimen that develops or improves a skill
 - Eliminate GOTOs
 - Eliminate Global Variables
 - Eliminate the "Useful and Flexible, but Unpredictable"

# The Functional Paradigm

 - Declarative
 - First Class Functions
 - Pure Functions
 - Immutability
 - Parallelism

# Declarative

 - "What" instead of "How"
 - Transformations of Data
 - Smart Compilers
 - Safety and Correctness

# First Class Functions

 - Functions as Fundamental Data Structures
 - Use Functions as Arguments to Other Functions
 - Abstract Common Patterns

# Pure Functions

 - Consistent, Predictible Results
 - No I/O or Modification of State
 - Safety
 - Optimization
 - Testing

# Pure Functions

Referential Transparency: Every occurance of a function *f* can be replaced with its return value without affecting the observable result of the program.

~~~~ {.java}
public int foo(int x) {

  // do some stuff

  launchMissiles(x);

  // do some more stuff...

  return x + 1;
}
~~~~

# Immutability

 - Can't Change Data
 - Immutable Structures can be Shared
 - e.g. Reusing Cons Cells / List Links

# Parallelism

 - Locks Unnecessary (less necessary)
 - Order of Execution negotiated by compiler

# Extra Stuff

 - Code Generation with Lisp
 - Type Safety with Haskell and OCaml
 - Massive Paralellism with Erlang
 - Haskell, Erlang, OCaml are all *fast*

# Part 2: Basic Haskell Syntax

# Basic Syntax

Basic Data Structures (nothing is an "object")

Numbers: All the Classics along with Ratios & Arbitrary Size/Precision Numbers

~~~~ {.haskell}
[1, 2, 3] == 1 : 2 : 3 : [] -- Lists
~~~~

~~~~ {.haskell}
[1..100], [10,9.5..0] -- Ranges
~~~~

~~~~ {.haskell}
"abcde" == ['a'..'e'] -- Strings are [Char]
~~~~

~~~~ {.haskell}
toUpper, sum, (+), (:) -- Functions
~~~~

~~~~ {.haskell}
(1, 3), ('T', [1,2,3]), (4, 'O', 4) -- Tuples
~~~~

~~~~ {.haskell}
True && False -- Booleans
~~~~

# Functions Definitions

~~~~ {.haskell}
twelve = 12

doubleMe x = x + x

even x = ((mod x 2) == 0)

odd x = (not (even x))

arbitrary f lst x = ((f x) : lst)
~~~~

# Algebraic Data Types

~~~~ {.haskell}
data Bool = True | False

data Color = Red | Green | Blue

data TrafficLight = Light [Char] [Char] Color
~~~~

# Type Parameters

Generic Datatypes

~~~~ {.haskell}
data Maybe a = Nothing | Just a

data Either a b = Left a | Right b
~~~~

# Part 3: Now with Types!

# Haskell's Type System

 - *Strong* & *Weak* Typing
 - *Static* & *Dynamic* Typing
 - Typecheck instead of "Run & Pray"
 - Type Inference & GHCi

# Function Signatures

~~~~ {.haskell}
12                 :: Int

[1, 2, 3]          :: [Int]

['f', 'o', 'o']    :: [Char]

"foo"              :: [Char]

(4, 'O', (Just 4)) :: (Int, Char, Maybe Int)

sum                :: [Int] -> Int
~~~~

# Quiz:

Reason by Function Signatures Alone

~~~~ {.haskell}
? :: Int -> Int -> Int
~~~~

# Quiz:

Reason by Function Signatures Alone

~~~~ {.haskell}
? :: Int -> Int -> Int
~~~~

~~~~ {.haskell}
? :: [Int] -> [Int]
~~~~

# Function Definitions

~~~~ {.haskell}
doubleMe :: Int -> Int
doubleMe x = x + x
~~~~

~~~~ {.haskell}
even :: Int -> Bool
even x = (mod x 2) == 0
~~~~

~~~~ {.haskell}
odd x = not (even x)
~~~~

# Type Constructors

~~~~ {.haskell}
data Maybe a = Nothing | Just a
~~~~

The parts before the "=" are *Type Constructors*.

e.g. Maybe Int, Maybe a

What we will normally refer to as the "type" of an expression.

# Value Constructors

~~~~ {.haskell}
data Bool = True | False
~~~~

The parts after the "=" are *Value Constructors*.

e.g. True and False

These are actually functions themselves.

~~~~ {.haskell}
False :: Bool

Just :: a -> Maybe a
~~~~



# Without Pattern Matching

~~~~ {.haskell}
isRed :: Color -> Bool
isRed c = if c == Red
          then True
          else if c == Blue...
~~~~

~~~~ {.haskell}
equals :: Color -> Color -> Bool
equals x y = if (x == Red && y == Red) || (x == Blue...
~~~~

# Pattern Matching

Pattern Matching: Disect data structures based on their value constructors.

~~~~ {.haskell}
data Color = Red | Green | Blue

show :: Color -> String
show Red = "Red"
show Green = "Green"
show Blue = "Blue"
~~~~

No "else" case needed!

~~~~ {.haskell}
(==) :: Color -> Color -> Bool
(==) Red Red     = True
(==) Green Green = True
(==) Blue Blue   = True
(==) x y         = False
~~~~


# Pattern Matching

~~~~ {.haskell}
data Maybe a = Nothing | Just a

val1 = Just 73
val2 = Nothing

possiblyDouble :: Maybe Int -> Maybe Int
possiblyDouble x = ...?
~~~~


# Pattern Matching

~~~~ {.haskell}
data Maybe a = Nothing | Just a

val1 = Just 73
val2 = Nothing

possiblyDouble :: Maybe Int -> Maybe Int
possiblyDouble Nothing  = Nothing
possiblyDouble (Just x) = Just (x + x)
~~~~



# Pattern Matching

Matching inside Lists

~~~~ {.haskell}
myList = [1,2,3,4,5]

sum :: [Int] -> Int
sum []     = 0
sum (x:xs) = x + sum xs
~~~~

# Case Statements

Matching with Case Statements

~~~~ {.haskell}
val1 = Just 73
val2 = Nothing

possiblyDouble :: Maybe Int -> Maybe Int
possibleDouble x = case x of
                   Nothing -> Nothing
                   (Just x) -> Just (x + x)
~~~~

# Guards

~~~~ {.haskell}
val1 = 5
val2 = 9001

doubleSmall :: Int -> Int
doubleSmall x
  | (abs x) <= 9000 = x + x
  | otherwise       = x
~~~~


# Typeclasses

The Problem: Function Scope

Equality for the Color type

~~~~ {.haskell}
(==) :: Color -> Color -> Bool
(==) colorA colorB = ...
~~~~

# Typeclasses

The Solution: Typeclasses

(Ad-Hoc Polymorphic Interfaces)

~~~~ {.haskell}
class Foo var where
  bar x :: a -> b
  baz x y :: a -> b -> c
  reFoo f :: var -> var
~~~~

# Typeclass Example

(Somewhere in the core libraries...)

~~~~ {.haskell}
class Eq a where
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool
~~~~

(Later on, in our file...)

~~~~ {.haskell}
instance Eq Color where
  (==) :: Color -> Color -> Bool
  (==) Red Red     = True
  (==) Green Green = True
  (==) Blue Blue   = True
  (==) _ _         = False

  (/=) :: Color -> Color -> Bool
  (/=) a b = not (a == b)
~~~~

# Part 4: Thinking Functionally

# To Understand Recursion...

~~~~ {.haskell}
myList = [1,2,3,4,5]

length :: [Int] -> Int
length [] = 0
length (x:xs) = 1 + length xs
~~~~

# Tail Recursion

~~~~ {.haskell}
myList = [1,2,3,4,5]

length :: [Int] -> Int
length lst = length' lst 0
       where length' [] len = len
             length' (x:xs) len = length' xs (len + 1)
~~~~

# Higher Order Functions

~~~~ {.haskell}
map :: (a -> b) -> [a] -> [b]

filter :: (a -> Bool) -> [a] -> [a]

zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]

foldl :: (a -> b -> a) -> a -> [b] -> a

foldl1 :: (a -> a -> a) -> [a] -> a

foldr :: (a -> b -> b) -> b -> [a] -> b

foldr1 :: (a -> a -> a) -> [a] -> a

any :: (a -> Bool) -> [a] -> Bool

all :: (a -> Bool) -> [a] -> Bool
~~~~

# Misc Syntax

~~~~ {.haskell}
-- Partial Application and Function Composition (.)
-- map :: (a -> b) -> [a] -> [b]
toNegative :: (Num a) => [a] -> [a]
toNegative = map (negate . abs)

-- Lambda
-- foldl :: (a -> b -> a) -> a -> [b] -> a
sumLarge = foldl (\acc x -> if x > 50 then acc + x else acc) 0 [1..100]

-- Function Application ($)
-- intersperse :: a -> [a] -> [a]
someNumbers = intersperse Nothing $ map Just [1..100]

-- Where
sumMaybes = foldl addMaybe 0 someNumbers
    where addMaybe Nothing  = acc
          addMaybe (Just v) = acc + v
~~~~

# Part 5: Something Practical

# Red Black Tree Overview
- Self Balancing Binary Search Trees
- O(log n) insert, search, delete

![](800px-Red-black_tree_example.png)

# Red Black Tree Data Structure

~~~~ {.haskell}
data Color = Red | Black deriving (Eq, Show)

data Tree a = Empty | Tree Color (Tree a) a (Tree a)
     deriving (Eq, Show)
~~~~

### Invariants

1. No Red node has a Red child.
1. Every path from the root to an Empty node contains the same number of Black nodes.

# Sample Tree

~~~~ {.haskell}
testTree1 = Tree Black Empty 100 Empty

testTree2 = Tree                                -- Value Constructor
                 Black                          -- color
                 (Tree Red Empty 50 Empty)      -- left
                 100                            -- value
                 (Tree Red Empty 150 Empty)     -- right
~~~~

# Simple Functions

~~~~ {.haskell}
isEmpty :: Tree a -> Bool
isEmpty Empty = True
isEmpty _ = False

height :: Tree a -> Int
height Empty = 0
height (Tree _ l _ r) = 1 + max (height l) (height r)

member :: (Ord a) => a -> Tree a -> Bool
member _ Empty = False
member x (Tree _ left elem right)
  | x < elem = member x left
  | x > elem = member x right
  | otherwise = True
~~~~

# Insert

~~~~ {.haskell}
insert :: (Ord a) => a -> Tree a -> Tree a
insert x Empty = Tree Black Empty x Empty
insert x s = let (Tree _ a y b) = ins s
             in Tree Black a y b
       where ins Empty = Tree Red Empty x Empty
             ins t@(Tree color a y b)
                 | x < y = balance (Tree color (ins a) y b)
                 | x > y = balance (Tree color a y (ins b))
                 | otherwise = t

fromList :: (Ord a) => [a] -> Tree a
fromList = foldl (flip insert) Empty

-- foldl === Ruby's inject
-- foldl :: (a -> b -> a) -> a -> [b] -> a
~~~~

# Balance

~~~~ {.haskell}
balance :: Tree a -> Tree a
balance (Tree Black (Tree Red (Tree Red a x b) y c) z d) = Tree Red (Tree Black a x b) y (Tree Black c z d)
balance (Tree Black (Tree Red a x (Tree Red b y c)) z d) = Tree Red (Tree Black a x b) y (Tree Black c z d)
balance (Tree Black a x (Tree Red (Tree Red b y c) z d)) = Tree Red (Tree Black a x b) y (Tree Black c z d)
balance (Tree Black a x (Tree Red b y (Tree Red c z d))) = Tree Red (Tree Black a x b) y (Tree Black c z d)
balance t = t
~~~~

# Balance

~~~~ {.haskell}
balance :: Tree a -> Tree a
balance (Tree Black (Tree Red (Tree Red a x b) y c) z d) =

balance (Tree Black (Tree Red a x (Tree Red b y c)) z d) =

balance (Tree Black a x (Tree Red (Tree Red b y c) z d)) =

balance (Tree Black a x (Tree Red b y (Tree Red c z d))) =

        Tree Red (Tree Black a x b) y (Tree Black c z d)

balance t = t
~~~~

# Map

~~~~ {.haskell}
isomap :: (a -> b) -> Tree a -> Tree b
isomap _ Empty = Empty
isomap f (Tree color a y b) = Tree color (isomap f a) (f y) (isomap f b)

nonisomap :: (Ord a, Ord b) => (a -> b) -> Tree a -> Tree b
nonisomap _ Empty = Empty
nonisomap f t = fromList $ map f $ toList t
~~~~


# Future Topics

- ``newtype'', ``type'', (\$), and Record Syntax
- Numeric Type Heirarchy and ByteStrings
- Parser Combinators
- Category Theory and Advanced Types
- Advanced Functional Data Structures
    + Zippers
    + Finger Trees (will blow your mind)
- Monads
- Common Monads (I/O, Reader, Writer, State)
- MonadTransformers and MonadPlus
- Arrows

# Future Topics (cont.)

- Testing/Quickcheck
- Error Handling
- Mutable Objects and Arrays
- Parallel Programming and STM
- Functional Reactive Programming
- Foreign Function Interface
- Cabal and Hackage
- Examine Real Code
    + XMonad Window Manager (<1000 lines)
    + Parsec
    + Yesod Web Framework
    + Darcs, Version Control System
    + The Glorious Glasgow Haskell Compiler
- Agda Theorem Prover


# Credits
- Sweet GPL Red Black Tree Diagram: en:User:Cburnett
- Slides format: Bryan O'Sullivan
- Numerous Examples: Learn You a Haskell for Great Good!
