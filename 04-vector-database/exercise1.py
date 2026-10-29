import chromadb

# 1. Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Get or create collection
collection = client.get_or_create_collection(name="recipes")

# 3. Define 15 recipes with metadata
recipes = [
    {"id": "r1", "title": "Spicy Thai Red Curry", "desc": "A rich, spicy coconut milk curry loaded with chilies, fresh garlic, lemongrass, and red curry paste.", "cuisine": "asian", "minutes": 25},
    {"id": "r2", "title": "Classic Margherita Pizza", "desc": "Traditional Italian thin-crust pizza topped with crushed San Marzano tomatoes, fresh mozzarella, and basil leaves.", "cuisine": "italian", "minutes": 20},
    {"id": "r3", "title": "Szechuan Mapo Tofu", "desc": "A hot and numbing Szechuan dish featuring soft tofu, minced pork, and pungent ground Szechuan peppercorns.", "cuisine": "asian", "minutes": 15},
    {"id": "r4", "title": "Creamy Fettuccine Alfredo", "desc": "Rich Italian pasta tossed in a velvety butter and parmesan cheese sauce with fresh ground black pepper.", "cuisine": "italian", "minutes": 15},
    {"id": "r5", "title": "Mexican Beef Tacos", "desc": "Warm seasoned ground beef inside crisp corn tortilla shells topped with hot salsa and jalapeno slices.", "cuisine": "mexican", "minutes": 20},
    {"id": "r6", "title": "Indian Chicken Tikka Masala", "desc": "Roasted tender chicken chunks in a warm, aromatic, lightly spiced tomato and cream gravy.", "cuisine": "indian", "minutes": 40},
    {"id": "r7", "title": "Japanese Tonkotsu Ramen", "desc": "Rich pork bone broth served hot with chewy noodles, soft-boiled marinated egg, and garlic oil.", "cuisine": "asian", "minutes": 60},
    {"id": "r8", "title": "Italian Minestrone Soup", "desc": "A warm, comforting vegetable soup brimming with beans, pasta, carrots, and tomatoes served hot.", "cuisine": "italian", "minutes": 35},
    {"id": "r9", "title": "Spicy Mexican Chilli Con Carne", "desc": "A simmering hot stew filled with ground beef, dark beans, hot chili peppers, and cumin.", "cuisine": "mexican", "minutes": 45},
    {"id": "r10", "title": "Greek Gyros Wrap", "desc": "Warm pita stuffed with seasoned roasted meat, fresh tzatziki sauce, sliced onions, and tomatoes.", "cuisine": "greek", "minutes": 15},
    {"id": "r11", "title": "Spicy Indian Chana Masala", "desc": "Fiery chickpea curry cooked with aromatic spices, fresh ginger, garlic, and hot green chilies.", "cuisine": "indian", "minutes": 30},
    {"id": "r12", "title": "Penne Arrabbiata", "desc": "Classic fiery Italian pasta dish cooked in a sharp, spicy tomato sauce with crushed red pepper flakes.", "cuisine": "italian", "minutes": 15},
    {"id": "r13", "title": "Vietnamese Hot Pho Soup", "desc": "Fragrant steamy beef noodle soup loaded with star anise, ginger, hot peppers, and fresh coriander.", "cuisine": "asian", "minutes": 50},
    {"id": "r14", "title": "Spanish Chicken Paella", "desc": "A warm rice dish cooked with saffron, bell peppers, chicken broth, and garlic over an open flame.", "cuisine": "spanish", "minutes": 45},
    {"id": "r15", "title": "Spicy Korean Kimchi Stew", "desc": "A hot, bubbling Korean stew made with aged kimchi, tofu, pork belly, and red pepper paste.", "cuisine": "asian", "minutes": 25},
]

# 4. Upsert recipes into the collection
collection.upsert(
    ids=[r["id"] for r in recipes],
    documents=[f"{r['title']}: {r['desc']}" for r in recipes],
    metadatas=[{"cuisine": r["cuisine"], "minutes": r["minutes"]} for r in recipes]
)

# 5. Perform query
query_text = "something warm and spicy"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

# 6. Display results
print(f"=== Top 3 results for query: '{query_text}' ===")
for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    print(f"- Recipe: {doc}")
    print(f"  Cuisine: {meta['cuisine']} | Prep time: {meta['minutes']} mins | Distance: {dist:.4f}\n")