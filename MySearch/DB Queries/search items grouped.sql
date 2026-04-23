 SELECT 
     regexp_replace(external_key, '_[^_]*$', '') AS folder_or_path,
     source,
     COUNT(*) as item_count
 FROM search_items
 GROUP BY folder_or_path, source
 ORDER BY 1 ASC;
