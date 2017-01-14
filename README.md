# haskell-cheatsheet
##Types
```haskell
type TempDiff = ((Int, Int), Double)
type Messwert = (Int, Double)

addThree :: Int -> Int -> Int -> Int  
addThree x y z = x + y + z  

ghci> :t (==)  
(==) :: (Eq a) => a -> a -> Bool 

ghci> :t (>)  
(>) :: (Ord a) => a -> a -> Bool 

data Op = Add | Sub
calc :: Op -> Int -> Int -> Int
calc Add a b = a + b
calc Sub a b = a - b

data Menu = BigMac | CheeseRoyal
data Size = Small | Large
type Order = (Menu,Size)
price :: Order -> Int
price (BigMac, Small) = 10 + 0;
price (BigMac, Large) = 10 + 2;
price (CheeseRoyal, Small) = 11 + 0;
price (CheeseRoyal, Large) = 11 + 2;
```
## Tuples
```haskell
ghci> fst (8,11)  --8 
ghci> snd (8,11)  --11  
```
## Lists
```haskell
ghci> [1,2,3,4] ++ [9,10,11,12]
ghci> "hello" ++ " " ++ "world"
ghci> 5:[1,2,3,4,5]  

--If you want to get an element out of a list by index, use !!. The indices start at 0.
ghci> [9.4,33.2,96.2,11.2,23.25] !! 1  --33.2
ghci> head [5,4,3,2,1]                 --5  
ghci> tail [5,4,3,2,1]                 --[4,3,2,1]
ghci> last [5,4,3,2,1]                 --1  
ghci> init [5,4,3,2,1]                 --[5,4,3,2]  
ghci> length [5,4,3,2,1]               --5 
ghci> null []                          --True 
take 3 [5,4,3,2,1]                     --[5,4,3] 
drop 3 [8,4,2,1,5,6]                   --[1,5,6]  
ghci> 4 `elem` [3,4,5,6]               --True  
```
##Präzedenz (absteigend)
```haskell
((f a) b) infixl
(.) infixr
(^) infixr
(*) (/) infixl
(+) (-) infixl
(++) (:) infixr
(==) (/=) (<) (>) (<=) (>=) infix (&&) infixr
(||) infixr
```
## Functions
```haskell
compareIf :: (Ord a) => a -> a -> Ordering
compareIf x y = if x < y
  then LT
  else if x == y
    then EQ
    else GT

compareGuard :: (Ord a) => a -> a -> Ordering
compareGuard x y | x < y = LT
                 | x == y = EQ
                 | otherwise = GT

compareCase :: (Ord a) => a -> a -> Ordering
compareCase x y = case x < y of
  True -> LT
  otherwise -> case x == y of
    True -> EQ
    otherwise -> GT
    
--
maximumG :: Int -> Int -> Int
maximumG a b | a >= b = a
             | a < b = b

maximumElse :: Int -> Int -> Int
maximumElse a b = if a >= b
                    then a
                    else b

maximumCase :: Int -> Int -> Int
maximumCase a b = case a >= b of
                    True -> a
                    False -> b    
```
### Pattern Matching
```haskell
data Currency = USD|EUR
toString :: Currency -> String
toString USD = "$"
toString EUR = "€"

first :: (a, b, c) -> a  
first (x, _, _) = x 

head' :: [a] -> a  
head' [] = error "Can't call head on an empty list, dummy!"  
head' (x:_) = x 
```
### if then else
```haskell
doubleSmallNumber x = if x > 100  
                        then x  
                        else x*2  
                       
doubleSmallNumber' x = (if x > 100 then x else x*2) + 1  
```
### Guards
```haskell
bmiTell :: (RealFloat a) => a -> a -> String  
bmiTell weight height  
    | weight / height ^ 2 <= 18.5 = "You're underweight, you emo, you!"  
    | weight / height ^ 2 <= 25.0 = "You're supposedly normal. Pffft, I bet you're ugly!"  
    | weight / height ^ 2 <= 30.0 = "You're fat! Lose some weight, fatty!"  
    | otherwise                 = "You're a whale, congratulations!"
    
max' :: (Ord a) => a -> a -> a  
max' a b   
    | a > b     = a  
    | otherwise = b  
```
### Where!?
```haskell
bmiTell :: (RealFloat a) => a -> a -> String  
bmiTell weight height  
    | bmi <= skinny = "You're underweight, you emo, you!"  
    | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"  
    | bmi <= fat    = "You're fat! Lose some weight, fatty!"  
    | otherwise     = "You're a whale, congratulations!"  
    where bmi = weight / height ^ 2  
          skinny = 18.5  
          normal = 25.0  
          fat = 30.0  
```
### Let it be
```haskell
cylinder :: (RealFloat a) => a -> a -> a  
cylinder r h = 
    let sideArea = 2 * pi * r * h  
        topArea = pi * r ^2  
    in  sideArea + 2 * topArea 
```
### Case
```haskell
head' :: [a] -> a  
head' xs = case xs of [] -> error "No head for empty lists!"  
                      (x:_) -> x  
```
### Curry
```haskell
addd :: (Int,Int) -> Int
addd (a,b) = a+b

currry :: ((a, b) -> c) -> a -> b -> c
currry f = \a b -> f (a,b)

resA = currry addd

--b) uncurry

adddd = currry addd;

uncurrry :: (a -> b -> c) -> (a,b) -> c
uncurrry f = \(a,b) -> f a b

resB = uncurrry adddd
```
## Recursion
```haskell
maximum' :: (Ord a) => [a] -> a  
maximum' [] = error "maximum of empty list"  
maximum' [x] = x  
maximum' (x:xs)   
    | x > maxTail = x  
    | otherwise = maxTail  
    where maxTail = maximum' xs 
    
maximum' :: (Ord a) => [a] -> a  
maximum' [] = error "maximum of empty list"  
maximum' [x] = x  
maximum' (x:xs) = max x (maximum' xs)  

replicate' :: (Num i, Ord i) => i -> a -> [a]  
replicate' n x  
    | n <= 0    = []  
    | otherwise = x:replicate' (n-1) x  
    
reverse' :: [a] -> [a]  
reverse' [] = []  
reverse' (x:xs) = reverse' xs ++ [x] 

zip' :: [a] -> [b] -> [(a,b)]  
zip' _ [] = []  
zip' [] _ = []  
zip' (x:xs) (y:ys) = (x,y):zip' xs ys  

elem' :: (Eq a) => a -> [a] -> Bool  
elem' a [] = False  
elem' a (x:xs)  
    | a == x    = True  
    | otherwise = a `elem'` xs   
    
quicksort :: (Ord a) => [a] -> [a]  
quicksort [] = []  
quicksort (x:xs) =   
    let smallerSorted = quicksort [a | a <- xs, a <= x]  
        biggerSorted = quicksort [a | a <- xs, a > x]  
    in  smallerSorted ++ [x] ++ biggerSorted  
    
deleteDuplicates :: (Eq a) => [a] -> [a]
deleteDuplicates [] = []
deleteDuplicates (a:list)
  | hasElement list a = deleteDuplicates list -- or elem'
  | otherwise = (deleteDuplicates list) ++ [a]
```
## Useful Functions
```haskell
ghci> zip [1,2,3,4,5] [5,5,5,5,5]  --[(1,5),(2,5),(3,5),(4,5),(5,5)]
ghci> show 3                       --"3"  
ghci> read "True" || False         --True 

(2+) interpreted as \y -> 2+y 
(+3) interpreted as \x -> x+3
```

```haskell
map :: (a -> b) -> [a] -> [b]
filter :: (a -> Bool) -> [a] -> [a]
zip :: [a] -> [b] -> [(a, b)]
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
drop :: Int -> [a] -> [a]
take :: Int -> [a] -> [a]
head :: [a] -> a
last :: [a] -> a
tail :: [a] -> [a]
init :: [a] -> [a]
reverse :: [a] -> [a]
length :: Foldable t => t a -> Int
concat :: Foldable t => t [a] -> [a]
elem :: (Foldable t, Eq a) => a -> t a -> Bool
and :: Foldable t => t Bool -> Bool
sum :: (Foldable t, Num a) => t a -> a
(!!) :: [a] -> Int -> a -- Get element by index
zipWith filter :: [a -> Bool] -> [[a]] -> [[a]]
tail.head.tail :: [[a]] -> [a]
```
## Examples
```haskell
import Data.Char

subjectSpam = "fuck credit is here"
subjectOk = "Hello World I am Bubu"

toLowerString :: String -> String
toLowerString s = map toLower s

normailze :: String -> [String]
normailze a = map toLowerString (filter (\x -> length x >= 3)(words a))

spam = ["viagra", "fuck", "credit", "hotel"]

rateWord :: String -> Int
rateWord s
  | (elem s spam) = -10
  | otherwise = 2

rateWords :: [String] -> Int
rateWords xs = sum (map rateWord xs)

--rateWords (normailze subjectOk)
--rateWords (normailze subjectSpam)

isSpam :: String -> Bool
isSpam xs = (rateWords (normailze xs)) < 0
-----------------------------------------------------
type Messwert = (Int, Double) -- (Woche, Temperatur)
type TempDiff = ((Int, Int), Double) -- ((Start Woche, End Woche), Temp. Differenz)

tempDB :: [Messwert]
tempDB = [(1, 5.6), (2, 4.8), (4, 5.9), (5, 4.2)]

tempDiffs :: [Messwert] -> [TempDiff]
tempDiffs (a:[]) = []
tempDiffs (l1:l2:list) = (\(a,b) -> \(c,d) -> [((a,c),d-b)]) l1 l2 ++ tempDiffs (l2:list)

findDiff :: Int -> [TempDiff]-> [Double]
findDiff _ [] = []
findDiff i (a:as)
  | i >= fst (fst a) && i < snd (fst a) = [snd a]
  | otherwise = findDiff i as

diffSumme:: Int -> Int -> [TempDiff] -> [Double]
diffSumme x y list = ds (findDiff x list) (findDiff y list)
  where ds [] _ = []
        ds _ [] = []
        ds (x:_) (y:_) = [x + y]
-----------------------------------------------------        
rep :: Int -> [Int] -> [Int]
rep a l = concat (map (\i -> if i == a
                                then [i,i]
                                else [i]) l)
```

```haskell
rem :: Int -> [Int] -> [Int]
rem a ls = concat (map (\x -> if x == a
                                then []
                                else [x]) ls)
```

```haskell
data Bit = Zero | One deriving (Eq)
toInt :: Bit -> Int
toInt a 
  | a == Zero = 0
  | otherwise = 1
```

```haskell
substring :: String -> Int -> Int -> String
substring [] _ _ = []
substring xs 0 0 = xs
substring [x:xs] o len = substring (tail xs) 0 (len-1)
substring [x:xs] from len = substring xs (from-1) len

```

# IO

## Quit Example
```haskell
main = do
  loop

loop = do
  putStrLn "Enter an expression or ':quit' to terminate"
  command <- getLine
  if command \= ":quit"
    then
      putStr "Result: " ++ (eval $ parseBExp command)
      loop
```

# DataTypes
```haskell
data Expr = Const Int
          | Add Expr Expr
          | Mul Expr Expr
          deriving (Show, Eq)
          
eval :: Expr -> Int
eval (Const n) = n
eval (Add l r) = eval l + eval r
eval (Mul l r) = eval l * eval r
```

```haskell
data Account = Account String [Mutation]
data Mutation = Deposit Int
              | Withdraw Int

instance Show Account where
  show (Account name ms) = name ++ " : " ++ (show (balance (Account name ms)))
```

# TypeClass
```haskell
data JSON = JSeq [JSON]
          | JObj [JBinding] 
          | JNum Double
          | JStr String
          | JBool Bool
          | JNull
          deriving (Show, Eq)
type JBinding = (String,JSON)
```

```haskell
class ToJSON a where
  toJSON :: a -> JSON

instance ToJSON Bool where
 toJSON b = JBool b

instance ToJSON Double where
  toJSON f = JNum f

instance ToJSON Int where
  toJSON i = JNum (fromIntegral i)
```

```haskell
instance Show Point where
  show (XY x y) = "P(x=" ++ show x ++ ",y=" ++ show y ++ ")"

instance Eq Point where
  (XY x1 y1) == (XY x2 y2) = x1 == x2 && y1 == y2

instance Ord Point where
  (XY x1 y1) <= (XY x2 y2) = (x1 < x2) || ((x1 == x2) && (y1 <= y2))

instance Eq Figure where
  (Circle p1 r1) == (Circle p2 r2) = p1 == p2 && r1 == r2
  (Line p1 p2)   == (Line q1 q2)   = p1 == q1 && p2 == q2
  _              == _              = False
```

## Modules
```haskell
module Geometry 
(circleArea 
,circlePerimeter 
,squareArea 
,squarePerimeter 
) where
```
