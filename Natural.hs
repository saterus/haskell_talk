import Data.Monoid

data Natural = Zero | Succ (Natural)

data Color = Red | Green | Blue
data TrafficLight = Light String String Color

instance Show Natural where
  show (Zero) = show 0
  show (Succ x) = show (count x 1)
    where count :: Natural -> Integer -> Integer
          count Zero sum = sum
          count (Succ n) sum = count n (1 + sum)

inc :: Natural -> Natural
inc x = Succ x
-- inc = Succ

dec :: Natural -> Natural
dec Zero = Zero
dec (Succ x) = x

plus :: Natural -> Natural -> Natural
plus (Zero) (Zero) = Zero
plus x (Zero) = x
plus (Zero) y = y
plus x (Succ y) = (plus (Succ x) y)

minus :: Natural -> Natural -> Natural
minus Zero Zero = Zero
minus Zero y = Zero
minus x Zero = x
minus (Succ x) (Succ y) = minus x y

instance Monoid Natural where
  mempty = Zero
  mappend x y = x `plus` y



