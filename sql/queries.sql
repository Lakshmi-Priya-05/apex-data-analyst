SELECT * FROM sales;

SELECT * FROM sales WHERE Gender='Female';

SELECT * FROM sales ORDER BY Sales DESC;

SELECT * FROM sales LIMIT 10;

SELECT COUNT(*) FROM sales;

SELECT SUM(Sales) FROM sales;

SELECT AVG(Sales) FROM sales;

SELECT MIN(Sales) FROM sales;

SELECT MAX(Sales) FROM sales;

SELECT "Product line", SUM(Sales)
FROM sales
GROUP BY "Product line";

SELECT City, SUM(Sales)
FROM sales
GROUP BY City;

SELECT Payment, COUNT(*)
FROM sales
GROUP BY Payment;

WITH AvgSales AS
(
SELECT AVG(Sales) avg_sales
FROM sales
)

SELECT *
FROM sales, AvgSales
WHERE Sales>avg_sales;

SELECT
"Invoice ID",
Sales,
RANK() OVER(ORDER BY Sales DESC)
FROM sales;

SELECT *
FROM HighSales;