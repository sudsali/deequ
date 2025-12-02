# Deequ Knowledge Base
*Comprehensive understanding of the Deequ library for AI bot context*

## Repository Overview
- **Purpose**: Unit tests for data quality on Apache Spark
- **Language**: Scala
- **Dependencies**: Apache Spark 3.1+, Java 8+
- **Package Structure**: com.amazon.deequ
- **License**: Apache 2.0

## Core Architecture

### Main Entry Point: VerificationSuite
- **Primary Class**: `com.amazon.deequ.VerificationSuite`
- **Purpose**: Main entry point for running data quality checks
- **Key Method**: `onData(data: DataFrame)` - starts verification process
- **Returns**: `VerificationRunBuilder` for fluent API construction

### Key Components:
1. **VerificationSuite** - Main orchestrator
2. **VerificationRunBuilder** - Fluent API builder pattern
3. **Check** - Individual data quality checks
4. **Analyzers** - Core computation engines
5. **MetricsRepository** - Storage for metrics
6. **AnalysisRunner** - Executes analysis

## Main Components

### 1. VerificationSuite (Main Entry)
- Responsible for running checks and analysis
- Returns verification results
- Integrates with metrics repositories
- Supports file output options

### 2. VerificationRunBuilder (Fluent API)
- **Purpose**: Builder pattern for constructing verification runs
- **Key Methods**:
  - `addCheck(check: Check)` - Add single check
  - `addChecks(checks: Seq[Check])` - Add multiple checks
  - `addRequiredAnalyzer()` - Force metric calculation
  - `useRepository()` - Enable metrics storage/reuse
  - `saveStatesWith()` - Save analyzer states for incremental computation
  - `aggregateWith()` - Load and aggregate previous states
  - `run()` - Execute the verification

### 3. Check Class (Data Quality Constraints)
- **Constructor**: `Check(level: CheckLevel.Value, description: String)`
- **CheckLevel**: Error, Warning
- **Returns**: `CheckWithLastConstraintFilterable` for most methods (allows filtering)

#### Core Constraint Methods:

**Size & Structure:**
- `hasSize(assertion: Long => Boolean)` - Assert on row count
- `hasColumnCount(assertion: Long => Boolean)` - Assert on column count
- `hasColumn(column: String)` - Check column exists

**Completeness (NULL checks):**
- `isComplete(column: String)` - Column has no NULLs
- `hasCompleteness(column, assertion: Double => Boolean)` - Custom completeness threshold
- `areComplete(columns: Seq[String])` - All columns complete
- `areAnyComplete(columns: Seq[String])` - At least one column complete

**Uniqueness:**
- `isUnique(column: String)` - Column values are unique
- `areUnique(columns: Seq[String])` - Combined columns are unique
- `isPrimaryKey(column: String, columns: String*)` - Primary key constraint
- `hasUniqueness(columns, assertion: Double => Boolean)` - Custom uniqueness threshold
- `hasDistinctness(columns, assertion: Double => Boolean)` - Distinct value ratio
- `hasUniqueValueRatio(columns, assertion: Double => Boolean)` - Unique value ratio

**Value Constraints:**
- `isContainedIn(column, allowedValues: Array[String])` - Values in allowed set
- `isNonNegative(column: String)` - No negative values
- `satisfies(expression: String, assertion: Double => Boolean)` - Custom SQL condition
- `customSql(expression: String, assertion: Double => Boolean)` - Custom SQL metric

**Pattern & Content:**
- `containsURL(column, assertion: Double => Boolean)` - URL pattern matching
- `containsEmail(column, assertion: Double => Boolean)` - Email pattern matching
- `matchesRegex(column, regex: Regex, assertion: Double => Boolean)` - Custom regex

**Statistical:**
- `hasApproxQuantile(column, quantile: Double, assertion: Double => Boolean)` - Quantile checks
- `hasMean(column, assertion: Double => Boolean)` - Mean value checks
- `hasStandardDeviation(column, assertion: Double => Boolean)` - Standard deviation
- `hasCorrelation(column1, column2, assertion: Double => Boolean)` - Column correlation

**Dataset Comparison:**
- `doesDatasetMatch(otherDataset, keyColumnMappings, assertion: Double => Boolean)` - Compare datasets

## Advanced Features

### 4. DQDL (Data Quality Definition Language)
- **Purpose**: Declarative language for defining data quality rules
- **Entry Point**: `EvaluateDataQuality.process(df, rulesetDefinition)`
- **Example**: `Rules=[IsUnique "item", RowCount < 10, Completeness "item" > 0.8]`

### 5. Analyzers (Core Computation Engines)
- **Base Trait**: `Analyzer[S <: State[_], +M <: Metric[_]]`
- **State Pattern**: Combinable states for incremental computation
- **Examples**: Completeness, Uniqueness, Size, Mean, StandardDeviation

### 6. Column Profiling
- **Entry Point**: `ColumnProfilerRunner().onData(df)`
- **Purpose**: Automatic data profiling and statistics generation

### 7. Constraint Suggestions
- **Entry Point**: `ConstraintSuggestionRunner().onData(df)`
- **Purpose**: Automatic constraint generation based on data patterns

### 8. Metrics Repository
- **Purpose**: Persist and reuse computed metrics
- **Implementations**: InMemory, FileSystem, Spark table storage

### 9. Anomaly Detection
- **Purpose**: Detect anomalies in metric time series
- **Strategies**: SimpleThreshold, BatchNormal, OnlineNormal, RateOfChange

## Common Usage Patterns

### Basic Example:
```scala
val result = VerificationSuite()
  .onData(df)
  .addCheck(
    Check(CheckLevel.Error, "integrity checks")
      .hasSize(_ == 5)
      .isComplete("id")
      .isUnique("id"))
  .run()
```

## Common Issues & Solutions

### 1. Spark Version Compatibility
- **Issue**: Deequ 2.x requires Spark 3.1+
- **Solution**: Use Deequ 1.x for older Spark versions

### 2. Memory Issues with Large Datasets
- **Solutions**: Use sampling, KLL sketches, configure Spark memory

### 3. Performance Optimization
- **Solutions**: Reuse metrics repository, combine checks, use approximate analyzers

---
*Knowledge Base Complete*
