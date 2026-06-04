# Movie Recommender System

---

## Overview

This project builds a movie recommender system using the MovieLens dataset enriched with IMDB data. It implements two content-based recommendation approaches and a collaborative filtering approach for personalized recommendations.

**Dataset:** MovieLens ml-latest-small (9,742 movies, 100,836 ratings, 610 users) + IMDB parquet data (9,739 entries)

---

## Project Structure

```
├── notebook.ipynb          # Main notebook with all tasks
├── movies.csv              # MovieLens movies
├── ratings.csv             # MovieLens ratings
├── tags.csv                # MovieLens user tags
├── links.csv               # MovieLens IMDB/TMDB links
└── imdb_data.parquet       # IMDB enrichment data
```

---

## Data Exploration & Cleaning

### Ratings

- 100,836 ratings from 610 users across 9,742 movies
- Ratings range 0.5 to 5.0. Mean and median are both 3.5 with a positive skew. This is **selection bias**: users tend to watch and rate movies they expect to enjoy, so low ratings are rare
- 4.0 is the most common rating. Whole star ratings are consistently more common than half star ratings
- **Long tail per movie:** median movie has only 3 ratings, mean is 10.4. 3,446 movies (35%) have just 1 rating. This sparsity is a key challenge for Task 3 since movies with very few ratings suffer from the **cold start problem**
- **Long tail per user:** median user rated 70 movies, mean is 165. One highly active user rated 2,698 movies. The dataset enforces a minimum of 20 ratings per user by design
- Timestamps span March 1996 to September 2018

### Genres

- 18 unique individual genres after excluding **IMAX** (not an official genre per the MovieLens README documentation, only listed implicitly through examples) and **(no genres listed)**
- Drama and Comedy dominate, appearing in ~4,300 and ~3,700 movies respectively. Film-Noir (87) and Western (167) are the rarest
- Average movie has 2.26 genres. 2,815 movies (29%) have only 1 genre, which produces weaker recommendations since there is less content signal
- **Drama + Comedy** is the most common genre pairing (1,012 movies). Documentary, Film-Noir and Western are the most isolated genres with few strong pairings
- **Film-Noir is the highest rated genre** (avg 3.92) despite being the rarest. Niche genres attract dedicated viewers. **Horror is the lowest** (3.26). The overall range is only ~0.65 which shows genre alone is not a strong predictor of quality

### Tags

- Only 58 of 610 users (9.5%) contributed tags, covering 1,572 movies (16% of dataset) with 1,589 unique tags
- Tags are highly diverse, ranging from actor names ("Leonardo DiCaprio") to themes ("drugs") to personal reactions ("way too long"). No consistent schema
- **Tagging was added later than ratings** (timestamps start 2006 vs 1996 for ratings), explaining lower user participation. This also means tags are biased toward more recent movies

### Data Quirk: IMDB ID Format

While investigating missing TMDB IDs, I discovered that IMDB IDs are stored as plain integers in the dataset, dropping leading zeros. Valid IMDB URLs require exactly 7 digits after `tt`, so IDs must be zero-padded:

```python
f"https://www.imdb.com/title/tt{str(imdbId).zfill(7)}/"
```

For example, imdbId `70820` becomes `tt0070820`. Verified by looking up Touki Bouki (1973), a 5-digit ID that needed two leading zeros. TMDB URLs use IDs directly with no padding needed: `https://www.themoviedb.org/movie/{tmdbId}`. This does not affect dataset joins since both files store IDs as plain integers.

### Data Cleaning Decisions

**Duplicate titles:** 5 pairs of movies with identical titles were found (e.g. Confessions of a Dangerous Mind, Emma, Eros, Saturn 3, War of the Worlds). Kept the version with more genre information. Duplicate detection was updated to use `subset=['title', 'year']` to correctly preserve legitimate same-title different-year films like Unforgiven (1992) and Unforgiven (2013).

**Wrong IMDB IDs:** Several MovieLens entries had IMDB IDs pointing to completely unrelated movies, discovered by manually checking IMDB pages after applying zero-padding:
- Death Note pointed to "Slika poslednje planete" (1970, Yugoslav documentary)
- Babylon 5 pointed to "Pluto's Fledgling" (1948, Disney short)
- Hyena Road pointed to Touki Bouki (1973), which already exists correctly in the dataset

**TV productions and non-movies dropped:**
- The Adventures of Sherlock Holmes and Doctor Watson, episode
- The OA, Cosmos, TV series
- Black Mirror (MovieLens entry), only genre was "short", not in our genre set
- Wallace & Gromit: The Best of Aardman Animation, compilation with no IMDB match
- Louis Theroux: Law & Disorder, TV documentary with no IMDB match
- Michael Jackson's Thriller, music video with no IMDB match
- **209 additional TV productions** discovered during IMDB enrichment join (miniseries, TV specials, OVAs that had slipped through under movie entries, see Task 2)

**Missing data handled:**
- 6 movies had missing years filled manually by referencing IMDB pages
- 5 movies had missing plot summaries filled manually from IMDB before the enrichment join
- 2 movies had missing genres filled from IMDB (Generation Iron 2 to Documentary, Maria Bamford: Old Baby to Documentary|Comedy)
- 3 movies with missing TMDB IDs filled manually after verifying on themoviedb.org, 3 others dropped (not found on TMDB, TV series, episode)

**Data type optimizations:** All ID columns downcast from int64 to int32, ratings from float64 to float32. Timestamps dropped from ratings and tags (not used as training features). Tags column converted from object to category. 32-bit types are sufficient for our value ranges and reduce memory usage.

**Final dataset shapes after all cleaning:**

| Table | Shape | Key columns |
|-------|-------|-------------|
| movies | (9,515, 4) | movieId, title, genres, year |
| ratings | (100,311, 3) | userId, movieId, rating |
| tags | (3,653, 3) | userId, movieId, tag |
| links | (9,515, 3) | movieId, imdbId, tmdbId |

---

## Task 1 — Content-Based Similar Movies

### Approach

Each movie is represented as a numeric vector of its features. Cosine similarity finds the most similar movies by measuring the angle between vectors.

**Features used:**
- 18 genre columns, multi-hot encoded using `MultiLabelBinarizer` (binary 0/1 per genre)
- Normalized year, `MinMaxScaler` to 0-1 scale

**Why cosine similarity:** Measures the angle between feature vectors rather than magnitude. Two movies with the same genre proportions are considered similar regardless of how many genres they have. For binary genre vectors this is the standard and most appropriate metric. Scores range from 0 (nothing in common) to 1 (identical).

**Why year normalization was added:** Genre-only similarity produced perfect 1.0 scores for many movies, any two films sharing the exact same genres were indistinguishable. Adding normalized year breaks these ties meaningfully.

Before year normalization, Annie Hall (1977, Comedy|Romance) recommended modern romcoms like How to Be Single (2016) and Crazy Stupid Love (2011). After adding normalized year, it correctly recommended era-appropriate films like Goodbye Girl (1977), Tootsie (1982) and Arthur (1981).

Similarly, Blade Runner (1982, Action|Sci-Fi|Thriller) was recommending Transformers (2007) and Iron Man 3 before year normalization. After, it recommended The Terminator (1984), Predator (1987) and other 1980s sci-fi thrillers.

**Core limitation:** Genre similarity struggles when genres are too broad. Schindler's List and Ran both have Drama|War but are completely different films, one is a Holocaust drama and the other is a Japanese feudal epic. This motivated Task 2.

### Recommendation Function

```python
get_similar_movies(title, n=10, year=None, year_range=None)
```

- Case-insensitive title matching
- **Disambiguation for duplicate titles**, when multiple movies share a title, prompts user to specify year: `get_similar_movies("Unforgiven", year=1992)`
- **"Did you mean" suggestions**, partial title matching for user errors (e.g. "Dark Knight" suggests Batman: The Dark Knight Returns)
- **Optional `year` parameter** for exact year specification
- **Optional `year_range` parameter** for era-specific recommendations

---

## Task 2 — IMDB Enrichment

### Approach

Enhanced Task 1 by joining IMDB data via `links.csv` using `imdbId` as the bridge key. A left join was used to keep all movies even if they have no IMDB match.

**Pre-filtering IMDB data:** Before joining, the IMDB parquet was filtered to only `titletype` values of `movie`, `tvMovie`, `video`, and `short`, removing the bulk of TV content upfront. The 209 TV productions removed during the join are entries that passed this filter but still had no match in our cleaned movies table.

**Important:** When combining multiple feature matrices, all matrices must have rows in the same order. Movie i in the genre matrix must correspond to movie i in the TF-IDF matrix. `movies_enriched` was used as the anchor DataFrame to ensure consistent alignment across all feature sources.

**New features added:**

**1. TF-IDF on plot summaries** (5,000 features, ngram_range=(1,2), stop_words='english', min_df=2, max_df=0.95)

TF-IDF (Term Frequency, Inverse Document Frequency) converts plot text into numeric vectors. Words appearing often in a specific plot but rarely across all plots get high scores, making distinctive words like "samurai" or "Holocaust" highly informative while common words like "the" are ignored. Using bigrams (ngram_range=(1,2)) captures meaningful phrases like "serial killer" that single words miss.

**2. Primary language** (25 one-hot encoded columns)

The raw language column in IMDB lists all languages a film was released in, including dubbed versions. Ran lists Japanese but also has France as a co-production country, technically still a Japanese film. It's a Wonderful Life lists both English and French, where the French is clearly a dub. Using the raw language list would treat dubs as equal to the original, which would hurt recommendations by grouping films by distribution language rather than cultural origin. The insight was that a film's true origin language is the one that matches the native language of its production country, so a country-to-native-language mapping could filter out dubs automatically. Dubbed languages are deliberately ignored: if a user wants the dubbed version they will know after being recommended the film and looking it up. No Python library cleanly maps ISO 3166 country codes to ISO 639 language codes for all 113 countries in the dataset, so the mapping was built manually.

A `get_primary_language()` function extracts the true origin language:
- Ran (countries: JP, FR, languages: ['ja']) correctly returns Japanese despite France co-financing
- Schindler's List (countries: US, languages: ['en', 'he', 'de', 'pl', 'la']) correctly returns English

**A nondeterminism bug was found and fixed:** The set intersection used to find matching native languages produced different results across kernel restarts due to Python's hash randomization (`list(set)[0]` is nondeterministic). Fixed by iterating through the original language array in order and returning the first match, preserving IMDB's original language ordering and making the pipeline fully deterministic.

**The `imdbrating` column** is stored as a nested dictionary, e.g. `{"rating": 8.1, "numberofvotes": 56039}`, rather than separate columns. Rating and vote count were extracted from this dict before use.

**3. Normalized IMDB rating** (quality signal, MinMaxScaler to 0-1)

Differentiates movies with identical genres by perceived quality. Blade Runner (8.1) vs Transformers (5.9) share Action|Sci-Fi|Thriller but are very different films.

**4. Normalized vote count** (reliability signal, log transform then MinMaxScaler)

A log transform was applied before scaling because vote counts have an extreme long tail (min 12, max 2.9 million). Raw MinMaxScaler would compress most values near 0. Log transform reflects how humans perceive popularity, the difference between 1 and 1,000 votes matters far more than the difference between 1 million and 2 million votes.

### Feature Combination with L2 Normalization

All feature groups were L2 normalized before combining into one matrix. Without this, TF-IDF (5,000 columns) would overwhelm genres (19 columns) purely due to dimensionality, genres would contribute almost nothing to the similarity score. L2 normalization ensures each feature group contributes equally regardless of column count.

Final combined matrix: 9,515 movies x 5,046 features (19 genres/year + 5,000 TF-IDF + 25 language + 1 rating + 1 votes)

### Key Finding: 209 TV Productions Removed

During the IMDB join, 209 MovieLens entries had no match in the IMDB data filtered to movie types. Investigation confirmed these were TV productions (miniseries, TV specials, OVAs, TV movies) that had slipped through the original MovieLens data under movie entries. These were identified and removed during the enrichment step, a data quality finding that only emerged through the IMDB integration.

### Improvement Demonstrated

| Query | Task 1 Result | Task 2 Result |
|-------|--------------|---------------|
| Schindler's List | Ran (Japanese feudal epic) appeared | Ran gone, Diary of Anne Frank, Patton, Flags of Our Fathers |
| Blade Runner | Transformers, Iron Man 3 appeared | Blade Runner 2049, Matrix, Terminator |
| Pulp Fiction | Fargo, In Bruges, already strong | Fargo, In Bruges, Snatch, still strong |

**Remaining limitation:** For very broad genre combinations like Comedy|Romance, TF-IDF plot vocabulary can still dominate. Annie Hall (1977) sometimes returns modern romcoms sharing plot vocabulary. The `year_range` parameter addresses this for users wanting era-specific results.

### Recommendation Function

```python
get_similar_movies_enriched(title, n=10, year=None, year_range=None)
```

Same interface as Task 1 for direct comparison.

---

## Task 3 — Personalized User Recommendations

### Approach: SVD (Singular Value Decomposition)

SVD is a matrix factorization technique that decomposes the user-movie rating matrix into latent factors, hidden characteristics that explain user preferences and movie properties. Unlike item-based collaborative filtering which fills missing ratings with 0 (introducing bias), SVD learns to predict missing ratings directly from discovered patterns in the data.

The full user-movie matrix is sparse, no user has rated every movie. SVD compresses it into two smaller matrices: a user matrix (each user gets a vector describing their taste profile) and a movie matrix (each movie gets a vector describing its characteristics). These factors are discovered automatically and unlabeled. To predict a rating, SVD takes the dot product of the user and movie vectors.

### Implementation

Uses the `scikit-surprise` library. Requires `numpy==1.26.4` (downgraded for compatibility).

**Data loading:** 100,311 ratings loaded into Surprise Dataset format with rating scale (0.5, 5.0).

**Train/test split:** 80/20, model learns from 80,248 ratings, evaluated on 20,063 held-out ratings.

**Model hyperparameters:**
- `n_factors=100`, number of latent factors (hidden taste/characteristic dimensions SVD discovers)
- `n_epochs=20`, passes through training data during gradient descent
- `lr_all=0.005`, learning rate, controls update step size
- `reg_all=0.02`, regularization, prevents overfitting by penalizing large factor values
- `random_state=42`, reproducibility

**Evaluation result: RMSE = 0.8747**

On average the model's predicted ratings are off by 0.87 stars. This is strong performance, the expected range for MovieLens SVD is 0.85 to 0.95.

### Recommendation Function

```python
get_recommendations(user_id, n=10)
```

For a given user, finds all movies they have not yet rated, predicts a rating for each one using the trained SVD model, sorts by predicted rating, and returns the top N with titles.

A `movie_id_to_title` dictionary is built upfront for O(1) lookups, faster than filtering the dataframe on every query.

### Results

**User 1, 232 ratings, heavy rater (124 at 5.0):**

Top recommendations include Shawshank Redemption, The Godfather, Rear Window, Dr. Strangelove, and Lawrence of Arabia, all critically acclaimed classics consistent with their rating history of MASH, Indiana Jones, American Beauty, and Goldfinger. All predicted at 5.0, reflecting how confident SVD is for a user with very consistent high-rating behavior.

**User 576, 19 ratings (minimum in dataset):**

Top recommendations include American History X (4.11), Blade Runner (4.05), In Bruges (4.02), Reservoir Dogs (3.95), and Apocalypse Now (3.95), dark gritty films matching their taste profile. The Saint and 13th Warrior rated highly, Reality Bites and Strictly Ballroom rated 1.0. The lower predicted score ceiling (4.11 vs 5.0) reflects the **cold start problem**: with only 19 ratings the model has less signal and is noticeably less confident, which shows directly in the predicted scores.

### Scalability Note

This implementation predicts ratings for every unrated movie per user, appropriate for ~9,500 movies. In production with millions of movies, approximate nearest neighbor search would replace brute-force prediction.

---

## Task 4 — LLM Tag Generation

### Approach

Built a small system that generates relevant tags for a movie given its title, year, and plot summary using Llama 3.3 70B via the Groq API. The system takes a plot summary as input and produces a comma-separated list of tags describing the movie's genre, themes, and notable aspects.

5 movies were chosen for testing based on having the most user tags in `tags.csv`, giving the richest ground truth to compare against: Pulp Fiction, Fight Club, Inception, Blade Runner, and Eternal Sunshine of the Spotless Mind. Plot summaries came from `movies_enriched` which already had them from the Task 2 IMDB join, so no extra loading was needed.

### Results

Note: LLM outputs can vary slightly between runs even at low temperature (temperature=0.3), so exact tags and precision numbers may differ from what is shown here if the notebook is re-run.

| Movie | LLM Tags | Exact Matches | Precision |
|-------|----------|---------------|-----------|
| Pulp Fiction | crime, drama, black comedy, non-linear narrative, violent, redemption, quirky, dark humor, neo-noir, intense, suspenseful, iconic dialogue | 8/12 | 67% |
| Fight Club | psychological thriller, dark comedy, subversive, anarchic, satirical, existential, rebellious, gritty, intense, thought-provoking, anti-capitalist, countercultural | 4/12 | 33% |
| Inception | action, sci-fi, thriller, mind-bending, psychological, complex, thought-provoking, futuristic, suspenseful, dark, intellectual | 5/11 | 45% |
| Blade Runner | science fiction, action, dystopian, thriller, philosophical, visually stunning, futuristic, cyberpunk, neo-noir, thought-provoking, classic | 2/11 | 18% |
| Eternal Sunshine | romance, drama, science fiction, psychological, memory loss, heartbreak, offbeat, thought-provoking, emotional, unique narrative, mind-bending | 3/11 | 27% |

**The precision numbers understate the actual quality.** Blade Runner scored 18% but the LLM generated "science fiction", "dystopian", "visually stunning", "neo-noir" and "futuristic", all clearly correct. Users tagged it "sci-fi" instead of "science fiction" and "future" instead of "futuristic". Same concept, different words. Eternal Sunshine scored 27% but "romance", "memory loss" and "heartbreak" are arguably the most precise possible tags for that film and none of them appear in the user tags. Users went with "bittersweet", "dreamlike", "arthouse", describing the feeling rather than the plot. Both are valid, just different vocabulary.

### Evaluation Discussion

Ground truth labels exist in `tags.csv`, 1,589 unique user-generated tags across 1,572 movies (16% of the dataset).

**Metrics that could be used:**
- **Exact match precision/recall**, strictest measure, counts only identical tags. Used above as a baseline
- **Semantic similarity**, using sentence embeddings to measure how close tags are in meaning space, handles synonyms ("scary" vs "frightening", "science fiction" vs "sci-fi")
- **BLEU/ROUGE**, standard NLP metrics for text generation quality

**Key challenges:**
- **Synonyms**, the biggest issue shown clearly in the results. "Science fiction" vs "sci-fi", "futuristic" vs "future", "dark humor" vs "dark comedy" are the same concept with different wording. Exact match treats them as completely different
- **Free-form user tags**, user tags include actor names ("Brad Pitt", "Chuck Palahniuk"), director names ("David Fincher"), source material ("based on a book") and personal reactions ("mindfuck"). The LLM generates thematic tags while users generate a mix of everything, so direct comparison has limits
- **Subjectivity**, there is no single correct set of tags for any movie. The LLM and different users would all produce valid but different tag sets
- **Coverage bias**, only 58 of 610 users contributed tags, and tagging started 10 years after ratings (2006 vs 1996), so ground truth is sparse and potentially unrepresentative

---

## What Worked Well

- Year normalization significantly improved era-appropriate recommendations, clearly demonstrated with Annie Hall and Blade Runner before/after comparisons
- Primary language extraction correctly handled co-productions and dubbed films using the country-native language mapping
- L2 normalization effectively balanced feature group contributions so genres weren't overwhelmed by TF-IDF dimensionality
- The nondeterminism bug in `get_primary_language` was caught through systematic testing across multiple kernel restarts
- The 209 TV productions discovery was an unexpected but valuable data quality finding that only emerged through the IMDB integration
- Duplicate title handling with year disambiguation produces a genuinely better user experience
- SVD achieved RMSE of 0.87, strong performance within the expected 0.85 to 0.95 range for MovieLens
- LLM tag generation produced accurate and relevant tags across all 5 test movies, with the low exact match precision reflecting a synonym mismatch problem rather than incorrect tags

## What Didn't Work as Well

- Annie Hall and films with broad genre labels still show imperfect era matching, TF-IDF plot vocabulary for romcoms is similar across decades
- Blade Runner recommendations still include some weaker matches, an explicit quality threshold using IMDB rating as a post-filter would help
- Feature weighting was approached through L2 normalization but explicit per-group weight tuning could further improve results
- User 1's all-5.0 predicted ratings show the model can be overconfident for users with very consistent high-rating behavior
- Exact match is too strict a metric for LLM tag evaluation, semantic similarity would be more appropriate

## Reflection

The hardest part was the IMDB data integration, the country-native language mapping required careful thought about co-productions, dubbed versions, and historical country codes (USSR, Czechoslovakia, Yugoslavia, East Germany all appear in the data). The nondeterminism bug in the primary language extraction cost significant debugging time because the symptom (fluctuating counts) appeared related to a different issue before the true cause was identified as Python's hash randomization on sets.

The data cleaning phase was more extensive than anticipated. 209 TV productions hidden in the MovieLens data, wrong IMDB IDs pointing to completely unrelated films, and the IMDB ID zero-padding discovery all emerged only through thorough investigation. This reinforced that understanding your data deeply before modeling is not optional, and that data quality issues often only surface when you try to join datasets from different sources.

SVD was straightforward to implement with the Surprise library, but the user analysis was the most interesting part, seeing how the model's confidence scales directly with the amount of data available per user made the cold start problem very concrete.

---

## References

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19.
