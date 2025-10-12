# NSCH Data Cleaning Script in R - DTA Only
# install.packages(c("haven", "dplyr"))
library(haven)
library(dplyr)

# Simple approach - assume we're running from the ProblemSet5 directory
cat("Current working directory:", getwd(), "\n")

# List what's in the current directory
cat("Files in current directory:\n")
print(list.files())

# Check if data directory exists
if (!dir.exists("data")) {
  cat("ERROR: 'data' directory not found in current directory\n")
  cat("Make sure you're running this from the ProblemSet5 directory\n")
  quit(save = "no", status = 1)
}

# Function to process a single year
process_year <- function(year) {
  cat("Processing year", year, "...")
  
  file_path <- file.path("data", paste0("nsch_", year, "e_topical.dta"))
  
  if (!file.exists(file_path)) {
    cat("MISSING\n")
    return(NULL)
  }
   
  df <- read_dta(file_path)
  df$year <- year
  
  cat(nrow(df), "rows\n")
  return(df)
}

# Process all years
years <- c(2016, 2017, 2019, 2020, 2021)
datasets <- list()

for (year in years) {
  df <- process_year(year)
  if (!is.null(df)) {
    datasets[[as.character(year)]] <- df
  }
}

if (length(datasets) > 0) {
  cat("\nCombining", length(datasets), "datasets...\n")
  
  # Convert problematic columns
  datasets <- lapply(datasets, function(df) {
    if ("stratum" %in% names(df)) {
      df$stratum <- as.character(df$stratum)
    }
    return(df)
  })
  
  # Combine all years
  all_data <- bind_rows(datasets)
  cat("Combined dataset:", nrow(all_data), "rows,", ncol(all_data), "columns\n")
  
  # Create output directory and save
  dir.create("cleaned_data", showWarnings = FALSE)
  write_dta(all_data, "cleaned_data/nsch_all_years_combined.dta")
  cat("SUCCESS: Saved to cleaned_data/nsch_all_years_combined.dta\n")
  
} else {
  cat("ERROR: No data files were found!\n")
}

cat("Done!\n")