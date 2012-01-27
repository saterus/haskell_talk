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

# Part 2: Basic Haskell

# Quick Tools Rundown

* GHC: The Glorious Glasgow Haskell Compilation System, version 7.0.3
* GHCi: GHC Interpreter
* Hackage: Haskell Package Database
* Cabal: Automated build, library, and dependency management tool
* Hoogle: Haskell Documentation Search
* Haskell.org: Community Website
* \#haskell on FreeNode: Very large, active IRC Channel

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
          else False
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

# Polymorphism

Parametric Polymorphism: a function does exactly the same thing
regardless of type.

~~~~ {.haskell}
id :: a -> a                    -- for all types a, return an a
map :: (a -> b) -> [a] -> [b]
~~~~

Ad Hoc Polymorphism: a function does different things based on the
type.

~~~~ {.haskell}
show :: a -> String  -- how do we know we can show this type?
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

# Using Typeclasses

~~~~ {.haskell}
(+) :: Num a => a -> a -> a

show :: Show a => a -> String
~~~~

Compiler can automatically derive instances of Read, Show, Bounded,
Enum, Eq, and Ord.

~~~~ {.haskell}
data Color = Red | Green | Blue
      deriving (Eq, Show, Read)
~~~~

# Modules

~~~~ {.haskell}
import Data.List (nub, sort)
import qualified Data.Map as M hiding(fold)
~~~~

~~~~ {.haskell}
module We.Use.Dots.To.Namespace.Crazy.Lib
( CrazyData(..)
, volume
, price
) where

-- implementation details
~~~~

See Hackage for more examples of the namespace heirarchy.

# Part 4: Thinking Functionally

# Local Binding

Where binds after/below use. Tends to be used for defining local functions.

~~~~ {.haskell}
sumMaybes = foldl maybeAdd 0 someNumbers
    where maybeAdd acc Nothing = acc
          maybeAdd acc (Just v) = acc + v
~~~~

Let creates them before/above use. Tends to be used for computing values.

~~~~ {.haskell}
minPayment price interestRate = let loanFee = 0.01
                                    interest = basePrice * interestRate
                                in loanFee + interest
~~~~

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

# Laziness

Automatically a lazy language. Expressions are evaluated only when
they are needed.

In effect, everything short circuits.

~~~~ {.haskell}
[1..] -- infinite list of integers

take 5 [1..]

times_ten x = x * 10

take 5 (map times_ten [1..])
~~~~

# Strictness

~~~~ {.haskell}
f !x = x + x

data Foo = !Int !String Char
~~~~

Sequence evaluation. Forces the evaluation the first arg, then the
second and return the result second. Strict for both args.

~~~~ {.haskell}
seq :: a -> b -> b
seq x y = ... -- must be a special compiler primitive.
~~~~

***Can sometimes screw up the compiler by ruining short cut fusion
optimizations that would otherwise be possible. Use carefully.***

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

# Map

~~~~ {.haskell}
map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs
~~~~


# Foldl

~~~~ {.haskell}
foldl f z []     = z
foldl f z (x:xs) = let z' = f z x
                   in foldl f z' xs

sum2 = foldl (+) 0

try2 = sum2 veryBigList
~~~~

![](left-fold.png)

# Foldr

~~~~ {.haskell}
foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ z []     =  z
foldr f z (x:xs) =  f x (foldr f z xs)

sum1 = foldr (+) 0

try1 = sum1 veryBigList
~~~~

![](right-fold.png)

# Foldr

~~~~ {.haskell}
foldr (+) 0 [1..1000000] -->
1 + (foldr (+) 0 [2..1000000]) -->
1 + (2 + (foldr (+) 0 [3..1000000])) -->
1 + (2 + (3 + (foldr (+) 0 [4..1000000]))) -->
1 + (2 + (3 + (4 + (foldr (+) 0 [5..1000000])))) -->
...
Stack Overflow!
~~~~

# Foldl

~~~~ {.haskell}
foldl (+) 0 [1..1000000] -->

let z1 =  0 + 1
in foldl (+) z1 [2..1000000] -->

let z1 =  0 + 1
    z2 = z1 + 2
in foldl (+) z2 [3..1000000] -->

let z1 =  0 + 1
    z2 = z1 + 2
    z3 = z2 + 3
in foldl (+) z3 [4..1000000] -->

let z1 =  0 + 1
    z2 = z1 + 2
    z3 = z2 + 3
    z4 = z3 + 4
in foldl (+) z4 [5..1000000] -->
...
Stack Overflow!
~~~~

# Foldl'

Fold Left Strict

~~~~ {.haskell}
seq :: a -> b -> b

foldl' :: (a -> b -> b) -> b -> [a] -> b
foldl' f z []     = z
foldl' f z (x:xs) = let z' = z `f` x
                    in seq z' $ foldl' f z' xs
~~~~

# Foldl'

~~~~ {.haskell}
foldl' (+) 0 [1..1000000] -->
foldl' (+) 1 [2..1000000] -->
foldl' (+) 3 [3..1000000] -->
foldl' (+) 6 [4..1000000] -->
foldl' (+) 10 [5..1000000] -->
~~~~

# Lambdas

~~~~ {.haskell}
map (\x -> x * x) [1..20]

sumLarge = foldl (\acc x -> if x > 50 then acc + x else acc) 0 [1..100]
~~~~

# Function Application

~~~~ {.haskell}
($) :: (a -> b) -> a -> b
f $ x = f x

-- intersperse :: a -> [a] -> [a]
someNumbers = intersperse Nothing $ map Just [1..100]
~~~~

# Partial Application

The traditional way to think of map:

~~~~ {.haskell}
map :: (a -> b) -> [a] -> [b]

map (\x -> x * x) [1..20]
~~~~

What if I only give it a function?

~~~~ {.haskell}
map :: (a -> b) -> ([a] -> [b])

map (\x -> x * x)
~~~~

Double Partial Application!

~~~~ {.haskell}
map (*10)
~~~~

# Currying

All Haskell functions only take 1 argument, partially apply it, and
then return more functions which take the next argument.

~~~~ {.haskell}
add :: (Num a) => a -> a -> a     -- these type defitions
add :: (Num a) => a -> (a -> a)   -- are the same
add arg1 arg2 = arg1 + arg2

plustwo :: (Num a) => a -> a
plustwo = add 2

five :: (Num a) => a
five = add 2 3                    -- functionally the same as
five = (add 2) 3                  -- using the parens

myFun :: a -> (b -> (c -> (d -> e)))
myFun w x y z = ...
~~~~

# Function Composition

Same Function Composition (f o g)(x) from high school algebra.

~~~~ {.haskell}
(.) :: (b -> c) -> (a -> b) -> (a -> c)
f . g = \x -> f (g x)

toNegative :: (Num a) => [a] -> [a]
toNegative = map (negate . abs)
~~~~

# Flip

Switch the order of the arguments of a function.

~~~~ {.haskell}
flip :: (a -> b -> c) -> (b -> a -> c)
flip f x y = f y x
~~~~

# Part 5: Intermission for Something Practical

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

# Tree Map

~~~~ {.haskell}
tmap :: (a -> b) -> Tree a -> Tree b
tmap _ Empty = Empty
tmap f (Tree color a y b) = Tree color (tmap f a) (f y) (tmap f b)
~~~~


# Tree Map

~~~~ {.haskell}
isomap :: (a -> b) -> Tree a -> Tree b
isomap _ Empty = Empty
isomap f (Tree color a y b) = Tree color (isomap f a) (f y) (isomap f b)

nonisomap :: (Ord a, Ord b) => (a -> b) -> Tree a -> Tree b
nonisomap _ Empty = Empty
nonisomap f t = fromList $ map f $ toList t
~~~~

# Smaller Practical Interlude B

Someone was wondering how a "pure" language like Haskell handles
I/O.


~~~~ {.haskell}
-- echo first argument

main :: IO ()
main = putStr (head getArgs)
~~~~

I'm not ready to explain how this works. Soon.

# Smaller Practical Interlude C

~~~~ {.haskell}
-- print out the response from an HTTP request

main :: IO ()
main = do
  (url:_) <- getArgs       -- Sets url to first command-line argument
  page <- simpleHttp url   -- Sets page to contents as a ByteString
  putStr (L.toString page) -- Converts ByteString to String and prints it
~~~~

Magic! But that is the style of the magic.

# Part 6: More Crazy Haskell!

# List Comprehensions

Write "Set-Builder" notation to create lists.

~~~~ {.haskell}
evens = [ n | n <- [1..], even n ]

squares = [ x * x | x <- [1..]]

powerset = [ (x,y) | x <- [1..4], y <- [1..4]]

variableName = [ this_var_goes_in_list | define_your_var, add_some_conditions, more_conditions ]
~~~~

# Record Syntax

~~~~ {.haskell}
data Config a b = Conf String String String String a String b String b

f :: Config Color [FilePath] -> Bool
f (Config _ _ _ _ _ p _ _ _) = hackEverything p
~~~~

What?

Better comment the shit out of that thing...

# Record Syntax

~~~~ {.haskell}
data Config a b = Conf { name :: String
                       , version :: String
                       , ip :: String
                       , count :: Int
                       , magic :: a
                       , password :: String
                       , incantationSequence :: b
                       , referrer :: String
                       , inCaseOfFire :: b }
~~~~

Creates functions for each field of the appropriate type.

~~~~ {.haskell}
-- generated
password :: Config a b -> String
password (Config _ _ _ _ _ p _ _ _) = p

f :: Config Color [FilePath] -> Bool
f (Config {password = p}) = hackEverything p
~~~~

# Defining DataTypes - type

Type Alias. Strictly a documentation convienence. A type "nickname".

~~~~ {.haskell}
type String = [Char]

type FilePath = String

type Rational = Ratio Integer
~~~~

Completely interchangable with the aliased "data" type.

Used almost everywhere a normal "data" type could be used (exception
being typeclass instance declaration).

# Defining DataTypes - newtype

Renames existing type and provides a new constructor. Stripped away at
compile time. Hide the underlying type.

~~~~ {.haskell}
newtype UniqueID = UniqueID Int deriving (Eq)
~~~~

No runtime overhead!

Exactly one field and exactly one constructor!

Not interchangable with the hidden/underlying "data" type. Compile Error!

Used everywhere a normal "data" type could be used (*including*
being typeclass instance declaration).


# Kinds

Kinds describe Types just like Types describe Values. We care about
how many type parameters are left to be locked in for this type.

We can use GHCi to query about the Kind of a Type. *'s correlate to type
parameters.

Roughly speaking, the type of a type.

~~~~ {.haskell}
-- Int is fully applied. No additional type parameters.
:k Int
Int :: *

-- Maybe isn't fully applied. Think of it as a type-function.
:k Maybe
Maybe :: * -> *

:k Maybe Int
Maybe Int :: *

:k Either
Either :: * -> * -> *

:k Either String
Either String :: * -> *
~~~~

Kinds are useful when thinking about higher order types.

# Part 7: Cool Abstractions

# Introduction to Abstraction

So let's review a couple of terms from Math class:

* **binary operation**: a function which takes two arguments. let's call it *.

* **identity**: an element, I, where I * a === a, or "Does not change our result"

* **associativity**: a * (b * c) === (a * b) * c, or "Order Applied"

* **commutativity**: a * b === b * a, or "Argument Order"

Why do you care?

# Monoids

First of the simple Haskell typeclasses named after scary Math stuff.

Almost stupidly simple.

~~~~ {.haskell}
class Monoid a where
  mempty :: a                 -- identity
  mappend :: a -> a -> a      -- an associative binary op
~~~~

So many types from all programming languages are Monoids. You just
didn't know it.

* Lists under concatenation (++)
* Numbers under Addition or Multiplication (+), (*)
* Booleans under Conjunction or Disjunction (all, any)
* Sets under Union or Intersection (||, &&)
* Functions from a type to itself, under Composition (.)

# Monoids

Monoid's type parameter, m, is a type of kind "*" (a fully applied type).

~~~~ {.haskell}
class Monoid m where
  mempty :: m                 -- identity
  mappend :: m -> m -> m      -- an associative binary op
  mconcat :: [m] -> m         -- optional, can be defined entirely in
                              -- terms of mappend otherwise
~~~~

Why do we care?

This is a pattern you see across many things since so many things
"qualify" as a Monoid.

# Monoids

~~~~ {.haskell}
class Monoid m where
  mempty :: m                 -- identity
  mappend :: m -> m -> m      -- an associative binary op
  mconcat :: [m] -> m         -- optional, can be defined entirely in
                              -- terms of mappend otherwise
~~~~

~~~~ {.haskell}
instance Monoid [] where
  mempty :: []
  mempty = []

  mappend :: m -> m -> m
  mappend = (++)

  mconcat :: [m] -> m
  mconcat = foldr mappend mempty
~~~~

# Monoid Laws

We can actually express the qualities of Monoids in Haskell notation,
which we can then test later using QuickCheck.

~~~~ {.haskell}
-- check the identity property of mempty
leftIdentity :: a -> Bool
leftIdentity x = mempty `mappend` x == x

rightIdentity :: a -> Bool
rightIdentity x = x `mappend` mempty = x

-- check associativity of mappend
associativity :: a -> a -> a -> Bool
associativity x y z = (x `mappend` y) `mappend` z == x `mappend` (y `mappend` z)
~~~~

That's it!

# Numbers

So Numbers are Monoids with using both the Addition and Multiplication
operations.

We can separate these by wrapping your number in a newtype wrapper.

~~~~ {.haskell}
newtype Sum a = Sum { getSum :: a }

newtype Product a = Product { getProduct :: a }
~~~~

We can then separately define Monoid instances for each of these
newtypes.

~~~~ {.haskell}
instance (Num a ) => (Monoid Sum) where
  mempty = Sum 0
  mappend (Sum x) (Sum y) = Sum $ x + y

instance (Num a ) => Monoid (Product a) where
  mempty = Sum 1
  mappend (Product x) (Product y) = Product $ x * y
~~~~

We can do the same thing for Bool's using newtype wrappers Any and All
for set conjunction and disjunction.

# Monoid Recap

The simplest of the (very) abstract type classes.

Usefulness? fold* can be defined in terms of "mempty" and "mappend".

~~~~ {.haskell}
foldl :: (Monoid a) => (a -> a -> a) -> a -> [a] -> a
~~~~

Very few requirements, all sorts of types are Monoids.

Can be extended in usefulness by defining CommutativeMonoids, which
requires commutativity of the type, where the result changes when
changing the argument order.

Commutative Monoids:

* Lists

* Functions

* Matrices

# Generalizing Map

Map takes a function and a "container", apply that function to the
value contained inside.

~~~~ {.haskell}
map          :: (a -> b) -> [a] -> [b]

tmap         :: (a -> b) -> Tree a -> Tree b

applyToMaybe :: (a -> b) -> Maybe a -> Maybe b

applyToLeft  :: (a -> b) -> Either a c -> Either b c
applyToRight :: (b -> c) -> Either a b -> Either a c

applyToLeft  :: (a -> b) -> (a,c) -> (b,c)
applyToRight :: (a -> b) -> (c,a) -> (c,b)
~~~~

More generally...

~~~~ {.haskell}
newtype Container t = Container t

mapContainer :: (a -> b) -> Container a -> Container b
~~~~

# Generalizing Map

Think of this as a tranformation of a function to operate in a new
context.

~~~~ {.haskell}
map          :: (a -> b) -> ([a] -> [b])

tmap         :: (a -> b) -> (Tree a -> Tree b)

applyToMaybe :: (a -> b) -> (Maybe a -> Maybe b)

applyToLeft  :: (a -> b) -> (Either a c -> Either b c)
applyToRight :: (a -> b) -> (Either c a -> Either c b)
~~~~

~~~~ {.haskell}
newtype Context t = Context t

mapContext :: (a -> b) -> (Context a -> Context b)
~~~~

You'll hear this referred to as ***Lifting: a concept which allows you
to transform a function into a corresponding function within another
setting.***


# Functors

Generalize this transformation into a typeclass called a Functor.

Functors are parameterized by f, which is also a type of kind "* -> *".

~~~~ {.haskell}
class Functor f where
  fmap :: Functor f => (a -> b) -> f a -> f b

instance Functor [] where
  fmap _ [] = []
  fmap g (x:xs) = g x : fmap g xs
~~~~

But why reimplement that? We can just assign the unapplied functions!

~~~~ {.haskell}
instance Functor [] where
  fmap = map

instance Functor Tree where
  fmap = tmap
~~~~

# Functor Usage

~~~~ {.haskell}
toUpper :: Char -> Char

capitalize :: (Functor f) => f Char -> f Char
capitalize = fmap toUpper

ghci> capitalize "abcedfg"
=> "ABCEDFG"
~~~~

# Functor Laws

Instances of Functor should satisfy the following laws:

Identity:

~~~~ {.haskell}
fmap id  ==  id
~~~~

Composition:

~~~~ {.haskell}
fmap (f . g)  ==  fmap f . fmap g
~~~~

Evil Functor Instance:

~~~~ {.haskell}
instance Functor [] where
  fmap _ [] = []
  fmap g (x:xs) = g x : g x : fmap g xs
~~~~

# Functors as Containers

Beyond lists: Functors as the container metaphor.

~~~~ {.haskell}
instance Functor Maybe where
  fmap _ Nothing = Nothing
  fmap g (Just x) = Just $ g x

instance Functor (Either e) where
  fmap _ (Left err) = Left err
  fmap g (Right x ) = Right $ g x
~~~~

Think of this as the partially applied type "(e,)". We needed a Kind
of "* -> *" for it to be a Functor instance.

~~~~ {.haskell}
instance Functor ((,) e) where
  fmap g (x,y) = (g x, y)
~~~~

# Functor Usage

~~~~ {.haskell}
possiblyDouble :: (Num a, Functor f) => f a -> f a
possiblyDouble = fmap (*2)

val1 = Just 12
val2 = Nothing

ghci> possiblyDouble val1
=> Just 24

ghci> possiblyDouble val2
=> Nothing

val3 = Left "DANGER!"
val4 = Right 404

ghci> possiblyDouble val3
=> Left "DANGER!"

ghci> possiblyDouble val4
=> Right 808

~~~~

# Functors as a Context

Functions are functors.   o.O

Haskell is all about viewing programs as "transformations of
data". Functions are first class data types. We can view Functors as
"transformations of functions".

Think of this type as "(e -> a), which is a function which takes an argument".

~~~~ {.haskell}
instance Functor ((->) e) where
  fmap :: ??
  fmap g h = \x -> g (h x)
~~~~

The type reveals all!

# Functors

Let's break this down.

~~~~ {.haskell}
instance Functor ((->) e) where
  fmap :: (b -> c) -> (a -> b) -> (a -> c)
  fmap g h = \x -> g (h x)
~~~~

Look's familar...

~~~~ {.haskell}
instance Functor ((->) e) where
  fmap :: (b -> c) -> (a -> b) -> (a -> c)
  fmap g h = g . h
~~~~

Why bother applying args in that case?

~~~~ {.haskell}
instance Functor ((->) e) where
  fmap :: (b -> c) -> (a -> b) -> (a -> c)
  fmap = (.)
~~~~

# Lists of Functions

~~~~ {.haskell}
fns :: [(Integer -> Integer)]
fns = map (*) [1..5]

-- alternatively written:

fns :: [(Integer -> Integer)]
fns = [(*1), (*2), (*3), (*4), (*5)]
~~~~

# Lists of Functions

~~~~ {.haskell}
fns :: [(Integer -> Integer)]
fns = map (*) [1..5]
~~~~

How do we get values back out of our container?

~~~~ {.haskell}
fns' = fmap (fmap (+3)) fns

:t fns'
=> ?
~~~~

# Lists of Functions

~~~~ {.haskell}
fns :: [(Integer -> Integer)]
fns = map (*) [1..5]
~~~~

How do we get values back out of our container?

~~~~ {.haskell}
fns' :: [(Integer -> Integer)]
fns' = fmap ?????? fns
~~~~

# Applicative Functors

Yet another typeclass abstracting away a pattern you didn't realize
existed!

~~~~ {.haskell}
class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
  (*>)  :: f a -> f a -> f b                     -- optional
  (<*)  :: f a -> f a -> f a                     -- optional
~~~~

(<*>) is pronounced "apply".

(*>) Sequence actions, discarding the value of the first argument.
(<*) Sequence actions, discarding the value of the second argument.

# Applicative Functor Example

First qualification: Maybe is a functor. Check!

Our Applicative Maybe will hold a function.

~~~~ {.haskell}
instance Applicative Maybe where

  pure :: a -> Maybe a
  pure = Just

  (<*>) :: Maybe (a -> b) -> Maybe a -> Maybe b
  Nothing <*> _ = Nothing
  (Just f) <*> x = fmap f x
~~~~

We get the other Applicative functions for free with the default
typeclass implementation, but we'll ignore them for now.

# Usage of Applicative Functors

Quiz Time!

~~~~ {.haskell}
ghci> pure (*10)
=> ?

ghci> pure (*10) <*> Just 12
=> ?

ghci> Nothing <*> Just 12
=> ?

ghci> pure (*10) <*> Nothing
=> ?

ghci> :t pure (*)
=> ?
~~~~

# Applicative Laws

~~~~ {.haskell}
-- Identity
pure id <*> v = v

-- Composition
pure (.) <*> u <*> v <*> w = u <*> (v <*> w)

-- Homomorphism
pure f <*> pure x = pure (f x)

-- Interchange
u <*> pure y = pure ($ y) <*> u
~~~~

# More Usage

Helpfully, the Applicative package (Control.Applicative) defines an
operator for our classic "fmap": <$>

~~~~ {.haskell}
nums = [1..10]

fns = fmap (*) nums

results = fns <*> [10]
~~~~

or we could write:

~~~~ {.haskell}
results = (*) <$> [1..10] <*> pure 10
~~~~

It looks funny now, but it grows on you. And you can start using it in
awesome ways.

See: Applicative Parsing.

# Monads

As Applicative Functors are more powerful than Functors, Monads are
more powerful than Applicative Functors.

As with Applicative Functors, Monads are parameterized by types of
kind "* -> *"

~~~~ {.haskell}
class Monad m where
  return :: a -> m a
  (>>=) :: m a -> (a -> m b) -> m b

  (>>) :: m a -> m b -> m b          -- optional, implemented by default
~~~~

(>>=) is pronounced "bind".

As we'll find out, many types are also Monads.

In fact, if it is a Monad, it is by necessity an Applicative Functor
(and we could define pure and (<*>) in terms of (>>=) and return).

# Maybe Monad

~~~~ {.haskell}
class Monad m where
  return :: a -> m a
  (>>=) :: m a -> (a -> m b) -> m b

  (>>) :: m a -> m b -> m b          -- optional, implemented by default
~~~~

~~~~ {.haskell}
instance Monad Maybe where
  return :: a -> Maybe a
  return = Just            -- again?

  (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b
  Nothing >>= _ = Nothing
  (Just x) >>= f = Just $ f x
~~~~

Similar semantics to our Applicative functions with Maybe. Why is this
cooler?

# More Maybe Monad

Let's see an example!

Applicative Style:

~~~~ {.haskell}
ghci> Just (+3) <*> Just 3
=> Just 6

ghci> Just (+3) <*> Nothing
=> Nothing
~~~~

Monadic Style:

~~~~ {.haskell}
ghci> Just 3 >>= (return . (+3))
~~~~

Hmm, sorta contrived example. Applicative looks better here.


# I can't believe we made it this far

I'm kinda out of slides. But not out of material!

# Better Maybe Monad example

Let's say we have a simple database (*cough*association-list*cough*).

~~~~ {.haskell}
newtype Database = Database [(String, String)]

db = Database [("Alice, "555-1234"), ("Bob", "555-9876"), ("Charlie", "555-3456")]

getPhoneNumber :: Database -> String -> Maybe String
getPhoneNumber (Database []) _ = Nothing
getPhoneNumber (Database ( (name,number) :xs) target
 | target == name = Just number
 | otherwise = getPhoneNumber xs target
~~~~

# Future Topics

- Common Monads (I/O, Reader, Writer, State)
- MonadTransformers and MonadPlus
- Arrows
- Numeric Type Heirarchy and ByteStrings
- Parser Combinators
- Category Theory and Advanced Types
- Advanced Functional Data Structures
    + Zippers
    + Finger Trees (will blow your mind)

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
- Fold Images: http://www.haskell.org/haskellwiki/Fold
- Slides format: Bryan O'Sullivan
- The Great Fold Examples:
- Learn You a Haskell for Great Good!
http://www.haskell.org/haskellwiki/Foldr_Foldl_Foldl%27
- Bryan O'Sullivan http://www.scs.stanford.edu/11au-cs240h/notes/
- Mark Lentczner
https://github.com/mtnviewmark/haskell-amuse-bouche/blob/master/slides.md

