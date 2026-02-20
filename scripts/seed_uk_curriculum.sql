-- ============================================================================
-- UK National Curriculum Seed Data
-- Primary School: Key Stage 1 (Year 1-2) and Key Stage 2 (Year 3-6)
-- Subjects: Mathematics, Geography, History
-- Source: UK Department for Education National Curriculum 2014
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. CURRICULUM
-- ============================================================================
INSERT INTO curricula (id, code, name, description, country_code, organization, version, year, is_active, extra_data)
VALUES (
    'c0000001-0000-0000-0000-000000000001',
    'uk_national_primary',
    'UK National Curriculum - Primary',
    'The National Curriculum for England sets out the programmes of study and attainment targets for all subjects at Key Stage 1 and Key Stage 2. It provides pupils with an introduction to the essential knowledge they need to be educated citizens.',
    'GB',
    'Department for Education',
    '2014',
    2014,
    true,
    '{"key_stages": ["KS1", "KS2"], "age_range": "5-11", "statutory": true}'::jsonb
);

-- ============================================================================
-- 2. GRADE LEVELS (Year 1-6)
-- ============================================================================
INSERT INTO grade_levels (id, curriculum_id, code, name, sequence, min_age, max_age) VALUES
    ('a0000001-0000-0000-0000-000000000001', 'c0000001-0000-0000-0000-000000000001', 'year_1', 'Year 1 (KS1)', 1, 5, 6),
    ('a0000001-0000-0000-0000-000000000002', 'c0000001-0000-0000-0000-000000000001', 'year_2', 'Year 2 (KS1)', 2, 6, 7),
    ('a0000001-0000-0000-0000-000000000003', 'c0000001-0000-0000-0000-000000000001', 'year_3', 'Year 3 (KS2)', 3, 7, 8),
    ('a0000001-0000-0000-0000-000000000004', 'c0000001-0000-0000-0000-000000000001', 'year_4', 'Year 4 (KS2)', 4, 8, 9),
    ('a0000001-0000-0000-0000-000000000005', 'c0000001-0000-0000-0000-000000000001', 'year_5', 'Year 5 (KS2)', 5, 9, 10),
    ('a0000001-0000-0000-0000-000000000006', 'c0000001-0000-0000-0000-000000000001', 'year_6', 'Year 6 (KS2)', 6, 10, 11);

-- ============================================================================
-- 3. SUBJECTS
-- ============================================================================
INSERT INTO subjects (id, curriculum_id, code, name, description, icon, color, sequence) VALUES
    ('50000001-0000-0000-0000-000000000001', 'c0000001-0000-0000-0000-000000000001', 'mathematics', 'Mathematics',
     'Mathematics is a creative and highly inter-connected discipline. The curriculum ensures pupils become fluent in fundamentals, reason mathematically, and solve problems.',
     'calculator', '#2196F3', 1),
    ('50000001-0000-0000-0000-000000000002', 'c0000001-0000-0000-0000-000000000001', 'geography', 'Geography',
     'Geography education inspires curiosity about the world and its people. Pupils gain knowledge about diverse places, people, resources, and natural and human environments.',
     'globe', '#4CAF50', 2),
    ('50000001-0000-0000-0000-000000000003', 'c0000001-0000-0000-0000-000000000001', 'history', 'History',
     'History education helps pupils gain coherent knowledge of Britain''s past and that of the wider world. It inspires curiosity to know more about the past.',
     'landmark', '#FF9800', 3);

-- ============================================================================
-- 4. UNITS - MATHEMATICS
-- ============================================================================

-- Year 1 Mathematics Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00101001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001',
     'y1_number_place_value', 'Number and Place Value',
     'Count to and across 100, identify one more/less, read and write numbers to 20 in numerals and words.', 1, 30),
    ('00101002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001',
     'y1_addition_subtraction', 'Addition and Subtraction',
     'Use number bonds within 20, add and subtract one-digit and two-digit numbers to 20.', 2, 35),
    ('00101003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001',
     'y1_multiplication_division', 'Multiplication and Division',
     'Solve simple problems using concrete objects, pictorial representations and arrays.', 3, 20),
    ('00101004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001',
     'y1_fractions', 'Fractions',
     'Recognise, find and name a half and a quarter of objects, shapes and quantities.', 4, 15),
    ('00101005-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001',
     'y1_measurement', 'Measurement',
     'Compare, describe and solve practical problems for lengths, mass, capacity and time.', 5, 25),
    ('00101006-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001',
     'y1_geometry', 'Geometry',
     'Recognise and name common 2-D and 3-D shapes, describe position, direction and movement.', 6, 20);

-- Year 2 Mathematics Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00102001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_number_place_value', 'Number and Place Value',
     'Count in steps of 2, 3, 5 from 0. Recognise place value in two-digit numbers. Compare and order numbers to 100.', 1, 30),
    ('00102002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_addition_subtraction', 'Addition and Subtraction',
     'Recall and use addition/subtraction facts to 20 fluently. Add and subtract using mental and written methods.', 2, 35),
    ('00102003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_multiplication_division', 'Multiplication and Division',
     'Recall and use multiplication/division facts for 2, 5 and 10 tables. Calculate using multiplication and division.', 3, 30),
    ('00102004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_fractions', 'Fractions',
     'Recognise and name fractions 1/3, 1/4, 2/4, 3/4. Recognise equivalence of 2/4 and 1/2.', 4, 20),
    ('00102005-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_measurement', 'Measurement',
     'Choose appropriate standard units for length, mass, capacity. Tell time to five minutes. Solve money problems.', 5, 30),
    ('00102006-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_geometry', 'Geometry',
     'Identify properties of 2-D and 3-D shapes. Describe position using right angles.', 6, 20),
    ('00102007-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002',
     'y2_statistics', 'Statistics',
     'Interpret and construct simple pictograms, tally charts and block diagrams.', 7, 15);

-- Year 3 Mathematics Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00103001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_number_place_value', 'Number and Place Value',
     'Count from 0 in multiples of 4, 8, 50, 100. Recognise place value in 3-digit numbers. Compare and order to 1000.', 1, 25),
    ('00103002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_addition_subtraction', 'Addition and Subtraction',
     'Add and subtract mentally with 3-digit numbers. Use columnar addition and subtraction methods.', 2, 30),
    ('00103003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_multiplication_division', 'Multiplication and Division',
     'Recall and use multiplication/division facts for 3, 4, 8 tables. Write statements using multiplication and division signs.', 3, 35),
    ('00103004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_fractions', 'Fractions',
     'Count up and down in tenths. Recognise and use fractions as numbers. Add and subtract fractions with same denominators.', 4, 25),
    ('00103005-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_measurement', 'Measurement',
     'Measure, compare lengths (m/cm/mm), mass (kg/g). Measure perimeter. Tell time using analogue clocks.', 5, 30),
    ('00103006-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_geometry', 'Geometry',
     'Recognise angles as properties of shape. Identify right angles, perpendicular and parallel lines.', 6, 20),
    ('00103007-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003',
     'y3_statistics', 'Statistics',
     'Interpret and present data using bar charts and pictograms.', 7, 15);

-- Year 4 Mathematics Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00104001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_number_place_value', 'Number and Place Value',
     'Count in multiples of 6, 7, 9, 25, 1000. Recognise place value in 4-digit numbers. Round to nearest 10, 100, 1000.', 1, 25),
    ('00104002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_addition_subtraction', 'Addition and Subtraction',
     'Add and subtract numbers with up to 4 digits using columnar methods.', 2, 25),
    ('00104003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_multiplication_division', 'Multiplication and Division',
     'Recall all multiplication tables up to 12x12. Use formal written methods for multiplication and division.', 3, 40),
    ('00104004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_fractions_decimals', 'Fractions and Decimals',
     'Recognise equivalent fractions. Count in hundredths. Recognise decimal equivalents of 1/4, 1/2, 3/4.', 4, 30),
    ('00104005-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_measurement', 'Measurement',
     'Convert between units. Calculate perimeter of rectilinear figures. Find area by counting squares.', 5, 30),
    ('00104006-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_geometry', 'Geometry',
     'Compare and classify shapes. Identify acute and obtuse angles. Identify lines of symmetry.', 6, 25),
    ('00104007-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004',
     'y4_statistics', 'Statistics',
     'Interpret and present data using bar charts and time graphs.', 7, 15);

-- Year 5 Mathematics Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00105001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_number_place_value', 'Number and Place Value',
     'Read, write, order and compare numbers to at least 1,000,000. Interpret negative numbers. Round to any degree.', 1, 25),
    ('00105002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_addition_subtraction', 'Addition and Subtraction',
     'Add and subtract with more than 4 digits using formal methods. Mental calculation with large numbers.', 2, 25),
    ('00105003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_multiplication_division', 'Multiplication and Division',
     'Identify multiples, factors, primes. Multiply up to 4 digits by 2 digits. Use short division. Square and cube numbers.', 3, 40),
    ('00105004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_fractions_decimals_percentages', 'Fractions, Decimals and Percentages',
     'Compare and order fractions. Add/subtract fractions. Multiply fractions by whole numbers. Recognise percentages.', 4, 35),
    ('00105005-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_measurement', 'Measurement',
     'Convert between metric units. Calculate perimeter and area. Estimate volume using 1cm cubed blocks.', 5, 30),
    ('00105006-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_geometry', 'Geometry',
     'Identify 3-D shapes from 2-D representations. Know angles in degrees. Draw and measure angles. Regular vs irregular polygons.', 6, 25),
    ('00105007-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000005',
     'y5_statistics', 'Statistics',
     'Solve problems using line graphs. Read and interpret tables including timetables.', 7, 15);

-- Year 6 Mathematics Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00106001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_number_place_value', 'Number and Place Value',
     'Read, write, order and compare numbers up to 10,000,000. Round to required accuracy. Use negative numbers in context.', 1, 20),
    ('00106002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_four_operations', 'Four Operations',
     'Long multiplication up to 4 digits by 2 digits. Long and short division. Mental calculations. Common factors and multiples.', 2, 35),
    ('00106003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_fractions_decimals_percentages', 'Fractions, Decimals and Percentages',
     'Simplify fractions. Add/subtract/multiply/divide fractions. Equivalences between fractions, decimals, percentages.', 3, 35),
    ('00106004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_ratio_proportion', 'Ratio and Proportion',
     'Solve problems involving relative sizes. Calculate percentages. Similar shapes and scale factors. Unequal sharing.', 4, 25),
    ('00106005-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_algebra', 'Algebra',
     'Use simple formulae. Generate and describe linear sequences. Express missing number problems algebraically.', 5, 25),
    ('00106006-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_measurement', 'Measurement',
     'Convert units including miles/kilometres. Calculate area of parallelograms and triangles. Volume of cuboids.', 6, 30),
    ('00106007-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_geometry', 'Geometry',
     'Draw 2-D shapes with given dimensions. Recognise 3-D shapes and nets. Classify shapes. Circle parts: radius, diameter, circumference.', 7, 25),
    ('00106008-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000006',
     'y6_statistics', 'Statistics',
     'Interpret and construct pie charts and line graphs. Calculate and interpret the mean.', 8, 20);

-- ============================================================================
-- 4. UNITS - GEOGRAPHY
-- ============================================================================

-- Year 1 Geography Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00201001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000001',
     'y1_locational_knowledge', 'Locational Knowledge',
     'Name and locate the world''s 7 continents and 5 oceans.', 1, 10),
    ('00201002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000001',
     'y1_uk_knowledge', 'United Kingdom',
     'Name, locate and identify characteristics of the 4 countries and capital cities of the United Kingdom.', 2, 12),
    ('00201003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000001',
     'y1_human_physical', 'Human and Physical Geography',
     'Identify seasonal and daily weather patterns. Identify physical features (beach, cliff, coast, forest, hill, mountain, sea, ocean, river, soil, valley, vegetation, season, weather) and human features (city, town, village, factory, farm, house, office, port, harbour, shop).', 3, 15),
    ('00201004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000001',
     'y1_fieldwork_skills', 'Geographical Skills and Fieldwork',
     'Use world maps, atlases and globes. Use simple compass directions and locational language. Use aerial photographs to recognise landmarks.', 4, 10);

-- Year 2 Geography Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00202001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000002',
     'y2_place_knowledge', 'Place Knowledge',
     'Understand geographical similarities and differences through studying a small area of the UK and a contrasting non-European country.', 1, 15),
    ('00202002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000002',
     'y2_hot_cold_places', 'Hot and Cold Places',
     'Identify the location of hot and cold areas of the world in relation to the Equator and the North and South Poles.', 2, 12),
    ('00202003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000002',
     'y2_fieldwork_mapping', 'Fieldwork and Mapping',
     'Use simple fieldwork and observational skills to study the geography of the school and its grounds.', 3, 10);

-- Year 3 Geography Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00203001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000003',
     'y3_europe_russia', 'Europe and Russia',
     'Locate European countries and major cities. Identify key physical and human characteristics of Europe.', 1, 15),
    ('00203002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000003',
     'y3_climate_zones', 'Climate Zones and Biomes',
     'Describe and understand key aspects of climate zones and biomes.', 2, 12),
    ('00203003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000003',
     'y3_uk_regions', 'UK Regions',
     'Name and locate counties and cities of the United Kingdom, geographical regions and their identifying characteristics.', 3, 15),
    ('00203004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000003',
     'y3_map_skills', 'Map Skills',
     'Use maps, atlases, globes and digital mapping to locate countries. Use the 8 points of a compass and 4-figure grid references.', 4, 10);

-- Year 4 Geography Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00204001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000004',
     'y4_americas', 'North and South America',
     'Locate and identify countries of North and South America, including environmental regions, key physical and human characteristics.', 1, 18),
    ('00204002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000004',
     'y4_rivers_water_cycle', 'Rivers and the Water Cycle',
     'Describe and understand key aspects of rivers and the water cycle.', 2, 15),
    ('00204003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000004',
     'y4_settlements_land_use', 'Settlements and Land Use',
     'Describe and understand key aspects of types of settlement and land use, economic activity including trade links.', 3, 12),
    ('00204004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000004',
     'y4_ordnance_survey', 'Ordnance Survey Maps',
     'Use Ordnance Survey maps to build knowledge of the UK and wider world. Use 6-figure grid references.', 4, 10);

-- Year 5 Geography Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00205001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000005',
     'y5_latitude_longitude', 'Latitude, Longitude and Time Zones',
     'Identify the position and significance of latitude, longitude, Equator, hemispheres, Tropics, Arctic/Antarctic Circles, Prime Meridian and time zones.', 1, 15),
    ('00205002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000005',
     'y5_mountains_volcanoes', 'Mountains, Volcanoes and Earthquakes',
     'Describe and understand key aspects of volcanoes and earthquakes, and mountains.', 2, 18),
    ('00205003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000005',
     'y5_european_region', 'European Region Study',
     'Understand geographical similarities and differences through the study of human and physical geography of a region in a European country.', 3, 15),
    ('00205004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000005',
     'y5_fieldwork_investigation', 'Fieldwork Investigation',
     'Use fieldwork to observe, measure, record and present human and physical features using a range of methods.', 4, 12);

-- Year 6 Geography Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00206001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000006',
     'y6_natural_resources', 'Natural Resources and Distribution',
     'Describe and understand key aspects of the distribution of natural resources including energy, food, minerals and water.', 1, 15),
    ('00206002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000006',
     'y6_american_region', 'American Region Study',
     'Understand geographical similarities and differences through the study of human and physical geography of a region in North or South America.', 2, 18),
    ('00206003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000006',
     'y6_global_trade', 'Global Trade and Economics',
     'Understand economic activity including trade links, and the distribution of natural resources.', 3, 12),
    ('00206004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000002', 'a0000001-0000-0000-0000-000000000006',
     'y6_digital_mapping', 'Digital Mapping and Analysis',
     'Use digital/computer mapping to locate countries and describe features. Present geographical information in a variety of ways.', 4, 10);

-- ============================================================================
-- 4. UNITS - HISTORY
-- ============================================================================

-- Year 1 History Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00301001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000001',
     'y1_living_memory', 'Changes Within Living Memory',
     'Learn about changes within living memory that reveal aspects of change in national life.', 1, 12),
    ('00301002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000001',
     'y1_significant_events', 'Significant Historical Events',
     'Learn about events beyond living memory that are significant nationally or globally, such as the Great Fire of London or the first aeroplane flight.', 2, 15),
    ('00301003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000001',
     'y1_significant_people', 'Significant Historical Figures',
     'Learn about the lives of significant individuals in the past who have contributed to national and international achievements.', 3, 12),
    ('00301004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000001',
     'y1_local_history', 'Local History',
     'Learn about significant historical events, people and places in own locality.', 4, 10);

-- Year 2 History Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00302001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000002',
     'y2_great_fire_london', 'The Great Fire of London',
     'Learn about events beyond living memory: the Great Fire of London (1666), its causes and effects on London.', 1, 15),
    ('00302002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000002',
     'y2_famous_explorers', 'Famous Explorers',
     'Compare the achievements of explorers: Christopher Columbus and Neil Armstrong - exploring new worlds.', 2, 12),
    ('00302003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000002',
     'y2_monarchs', 'British Monarchs',
     'Compare the lives of significant individuals: Elizabeth I and Queen Victoria.', 3, 15),
    ('00302004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000002',
     'y2_nursing_pioneers', 'Nursing Pioneers',
     'Learn about the achievements of Florence Nightingale and Mary Seacole.', 4, 10);

-- Year 3 History Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00303001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000003',
     'y3_stone_age', 'Stone Age to Iron Age Britain',
     'Learn about changes in Britain from the Stone Age to the Iron Age including late Neolithic hunter-gatherers and early farmers.', 1, 20),
    ('00303002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000003',
     'y3_bronze_age', 'Bronze Age Britain',
     'Learn about Bronze Age religion, technology and travel including Stonehenge.', 2, 15),
    ('00303003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000003',
     'y3_iron_age', 'Iron Age Britain',
     'Learn about Iron Age hill forts, tribal kingdoms, farming, art and culture.', 3, 15);

-- Year 4 History Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00304001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000004',
     'y4_roman_empire', 'The Roman Empire and its Impact on Britain',
     'Learn about Julius Caesar''s attempted invasion, the Roman Empire and its army, successful invasion by Claudius and conquest.', 1, 20),
    ('00304002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000004',
     'y4_roman_britain', 'Roman Britain',
     'Learn about British resistance including Boudica, Romanisation of Britain: roads, towns, architecture, technology, beliefs.', 2, 18),
    ('00304003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000004',
     'y4_ancient_egypt', 'Ancient Egypt',
     'Learn about the achievements of one of the earliest civilizations: Ancient Egypt and the pyramids, pharaohs, and daily life.', 3, 18);

-- Year 5 History Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00305001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000005',
     'y5_anglo_saxons', 'Anglo-Saxon Britain',
     'Learn about Britain''s settlement by Anglo-Saxons: invasions, settlements, kingdoms, place names, village life, art and culture.', 1, 18),
    ('00305002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000005',
     'y5_vikings', 'The Viking and Anglo-Saxon Struggle',
     'Learn about Viking raids and invasion, resistance by Alfred the Great, Danegeld and further Viking invasions.', 2, 18),
    ('00305003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000005',
     'y5_ancient_greece', 'Ancient Greece',
     'Learn about Ancient Greek life and achievements including democracy, philosophy, art, architecture, Olympic Games, and their influence on the Western world.', 3, 20);

-- Year 6 History Units
INSERT INTO units (id, subject_id, grade_level_id, code, name, description, sequence, estimated_hours) VALUES
    ('00306001-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000006',
     'y6_edward_confessor', 'Edward the Confessor and 1066',
     'Learn about Edward the Confessor and his death in 1066, leading to the Norman Conquest.', 1, 15),
    ('00306002-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000006',
     'y6_mayan_civilization', 'Mayan Civilization',
     'Learn about a non-European society that provides contrasts with British history: the Mayan civilization c. AD 900.', 2, 18),
    ('00306003-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000006',
     'y6_world_war_two', 'Britain Since 1930: World War Two',
     'Learn about a significant turning point in British history: World War Two and its impact on Britain and the world.', 3, 20),
    ('00306004-0000-0000-0000-000000000001', '50000001-0000-0000-0000-000000000003', 'a0000001-0000-0000-0000-000000000006',
     'y6_local_history_depth', 'Local History Study',
     'A local history study examining an aspect of history or a site dating from a period beyond 1066.', 4, 12);

-- ============================================================================
-- 5. TOPICS - MATHEMATICS YEAR 1
-- ============================================================================

-- Y1 Number and Place Value Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70101001-0001-0000-0000-000000000001', '00101001-0000-0000-0000-000000000001', 'y1_counting_100', 'Counting to 100',
     'Count to and across 100, forwards and backwards, beginning with 0 or 1, or from any given number.', 1, 0.30, 60),
    ('70101001-0002-0000-0000-000000000001', '00101001-0000-0000-0000-000000000001', 'y1_counting_multiples', 'Counting in Multiples',
     'Count, read and write numbers to 100 in numerals; count in multiples of 2s, 5s and 10s.', 2, 0.35, 60),
    ('70101001-0003-0000-0000-000000000001', '00101001-0000-0000-0000-000000000001', 'y1_one_more_less', 'One More and One Less',
     'Given a number, identify one more and one less.', 3, 0.30, 45),
    ('70101001-0004-0000-0000-000000000001', '00101001-0000-0000-0000-000000000001', 'y1_number_words', 'Reading and Writing Numbers',
     'Read and write numbers from 1 to 20 in numerals and words.', 4, 0.35, 60);

-- Y1 Addition and Subtraction Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70101002-0001-0000-0000-000000000001', '00101002-0000-0000-0000-000000000001', 'y1_number_bonds', 'Number Bonds',
     'Represent and use number bonds and related subtraction facts within 20.', 1, 0.35, 90),
    ('70101002-0002-0000-0000-000000000001', '00101002-0000-0000-0000-000000000001', 'y1_add_subtract_20', 'Adding and Subtracting to 20',
     'Read, write and interpret mathematical statements involving addition (+), subtraction (-) and equals (=) signs.', 2, 0.40, 90),
    ('70101002-0003-0000-0000-000000000001', '00101002-0000-0000-0000-000000000001', 'y1_add_one_two_digit', 'Adding One-Digit and Two-Digit Numbers',
     'Add and subtract one-digit and two-digit numbers to 20, including zero.', 3, 0.45, 90),
    ('70101002-0004-0000-0000-000000000001', '00101002-0000-0000-0000-000000000001', 'y1_missing_number', 'Missing Number Problems',
     'Solve one-step problems that involve addition and subtraction, using concrete objects and pictorial representations.', 4, 0.45, 60);

-- Y1 Multiplication and Division Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70101003-0001-0000-0000-000000000001', '00101003-0000-0000-0000-000000000001', 'y1_grouping', 'Grouping and Sharing',
     'Solve one-step problems involving multiplication and division, by calculating the answer using concrete objects, pictorial representations and arrays.', 1, 0.40, 60),
    ('70101003-0002-0000-0000-000000000001', '00101003-0000-0000-0000-000000000001', 'y1_doubling', 'Doubling',
     'Understand doubling as adding two equal groups.', 2, 0.35, 45),
    ('70101003-0003-0000-0000-000000000001', '00101003-0000-0000-0000-000000000001', 'y1_halving', 'Halving',
     'Understand halving as sharing into two equal groups.', 3, 0.35, 45);

-- Y1 Fractions Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70101004-0001-0000-0000-000000000001', '00101004-0000-0000-0000-000000000001', 'y1_half', 'Recognising Halves',
     'Recognise, find and name a half as one of two equal parts of an object, shape or quantity.', 1, 0.40, 45),
    ('70101004-0002-0000-0000-000000000001', '00101004-0000-0000-0000-000000000001', 'y1_quarter', 'Recognising Quarters',
     'Recognise, find and name a quarter as one of four equal parts of an object, shape or quantity.', 2, 0.45, 45);

-- Y1 Measurement Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70101005-0001-0000-0000-000000000001', '00101005-0000-0000-0000-000000000001', 'y1_length_height', 'Length and Height',
     'Compare, describe and solve practical problems for lengths and heights.', 1, 0.35, 45),
    ('70101005-0002-0000-0000-000000000001', '00101005-0000-0000-0000-000000000001', 'y1_mass_weight', 'Mass and Weight',
     'Compare, describe and solve practical problems for mass/weight.', 2, 0.35, 45),
    ('70101005-0003-0000-0000-000000000001', '00101005-0000-0000-0000-000000000001', 'y1_capacity_volume', 'Capacity and Volume',
     'Compare, describe and solve practical problems for capacity and volume.', 3, 0.35, 45),
    ('70101005-0004-0000-0000-000000000001', '00101005-0000-0000-0000-000000000001', 'y1_time', 'Time',
     'Sequence events in chronological order. Tell the time to the hour and half past the hour.', 4, 0.40, 60);

-- Y1 Geometry Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70101006-0001-0000-0000-000000000001', '00101006-0000-0000-0000-000000000001', 'y1_2d_shapes', '2-D Shapes',
     'Recognise and name common 2-D shapes including rectangles, squares, circles and triangles.', 1, 0.30, 45),
    ('70101006-0002-0000-0000-000000000001', '00101006-0000-0000-0000-000000000001', 'y1_3d_shapes', '3-D Shapes',
     'Recognise and name common 3-D shapes including cuboids, cubes, pyramids and spheres.', 2, 0.35, 45),
    ('70101006-0003-0000-0000-000000000001', '00101006-0000-0000-0000-000000000001', 'y1_position_direction', 'Position and Direction',
     'Describe position, direction and movement, including whole, half, quarter and three-quarter turns.', 3, 0.40, 45);

-- ============================================================================
-- 5. TOPICS - MATHEMATICS YEAR 2
-- ============================================================================

-- Y2 Number and Place Value Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102001-0001-0000-0000-000000000001', '00102001-0000-0000-0000-000000000001', 'y2_counting_steps', 'Counting in Steps',
     'Count in steps of 2, 3, and 5 from 0, and in tens from any number, forward and backward.', 1, 0.35, 60),
    ('70102001-0002-0000-0000-000000000001', '00102001-0000-0000-0000-000000000001', 'y2_place_value', 'Place Value in Two-Digit Numbers',
     'Recognise the place value of each digit in a two-digit number (tens, ones).', 2, 0.40, 60),
    ('70102001-0003-0000-0000-000000000001', '00102001-0000-0000-0000-000000000001', 'y2_compare_order', 'Compare and Order Numbers',
     'Compare and order numbers from 0 up to 100; use < > and = signs.', 3, 0.40, 45),
    ('70102001-0004-0000-0000-000000000001', '00102001-0000-0000-0000-000000000001', 'y2_number_problems', 'Number Problems',
     'Use place value and number facts to solve problems.', 4, 0.45, 60);

-- Y2 Addition and Subtraction Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102002-0001-0000-0000-000000000001', '00102002-0000-0000-0000-000000000001', 'y2_mental_addition', 'Mental Addition',
     'Add numbers using concrete objects, pictorial representations, and mentally including two-digit numbers.', 1, 0.40, 60),
    ('70102002-0002-0000-0000-000000000001', '00102002-0000-0000-0000-000000000001', 'y2_mental_subtraction', 'Mental Subtraction',
     'Subtract numbers using concrete objects, pictorial representations, and mentally including two-digit numbers.', 2, 0.40, 60),
    ('70102002-0003-0000-0000-000000000001', '00102002-0000-0000-0000-000000000001', 'y2_inverse_operations', 'Inverse Operations',
     'Recognise and use the inverse relationship between addition and subtraction to check calculations.', 3, 0.45, 45),
    ('70102002-0004-0000-0000-000000000001', '00102002-0000-0000-0000-000000000001', 'y2_word_problems', 'Addition and Subtraction Word Problems',
     'Solve problems with addition and subtraction using concrete objects and pictorial representations.', 4, 0.50, 60);

-- Y2 Multiplication and Division Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102003-0001-0000-0000-000000000001', '00102003-0000-0000-0000-000000000001', 'y2_2_times_table', '2 Times Table',
     'Recall and use multiplication and division facts for the 2 multiplication table.', 1, 0.40, 60),
    ('70102003-0002-0000-0000-000000000001', '00102003-0000-0000-0000-000000000001', 'y2_5_times_table', '5 Times Table',
     'Recall and use multiplication and division facts for the 5 multiplication table.', 2, 0.40, 60),
    ('70102003-0003-0000-0000-000000000001', '00102003-0000-0000-0000-000000000001', 'y2_10_times_table', '10 Times Table',
     'Recall and use multiplication and division facts for the 10 multiplication table.', 3, 0.35, 45),
    ('70102003-0004-0000-0000-000000000001', '00102003-0000-0000-0000-000000000001', 'y2_multiplication_problems', 'Multiplication and Division Problems',
     'Solve problems involving multiplication and division using materials, arrays, repeated addition and mental methods.', 4, 0.50, 60);

-- Y2 Fractions Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102004-0001-0000-0000-000000000001', '00102004-0000-0000-0000-000000000001', 'y2_unit_fractions', 'Unit Fractions',
     'Recognise, find, name and write fractions 1/3, 1/4, 2/4 and 3/4 of a length, shape, set of objects or quantity.', 1, 0.45, 60),
    ('70102004-0002-0000-0000-000000000001', '00102004-0000-0000-0000-000000000001', 'y2_equivalent_fractions', 'Simple Equivalent Fractions',
     'Write simple fractions and recognise the equivalence of 2/4 and 1/2.', 2, 0.50, 45);

-- Y2 Measurement Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102005-0001-0000-0000-000000000001', '00102005-0000-0000-0000-000000000001', 'y2_standard_units', 'Standard Units',
     'Choose and use appropriate standard units to estimate and measure length/height (m/cm), mass (kg/g), temperature (°C), capacity (litres/ml).', 1, 0.45, 60),
    ('70102005-0002-0000-0000-000000000001', '00102005-0000-0000-0000-000000000001', 'y2_money', 'Money',
     'Recognise and use symbols for pounds (£) and pence (p); combine amounts to make a particular value.', 2, 0.45, 60),
    ('70102005-0003-0000-0000-000000000001', '00102005-0000-0000-0000-000000000001', 'y2_time_5_minutes', 'Telling Time to 5 Minutes',
     'Tell and write the time to five minutes, including quarter past/to the hour.', 3, 0.50, 60);

-- Y2 Geometry Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102006-0001-0000-0000-000000000001', '00102006-0000-0000-0000-000000000001', 'y2_shape_properties', 'Properties of Shapes',
     'Identify and describe the properties of 2-D shapes, including number of sides and line symmetry.', 1, 0.40, 45),
    ('70102006-0002-0000-0000-000000000001', '00102006-0000-0000-0000-000000000001', 'y2_3d_properties', '3-D Shape Properties',
     'Identify and describe the properties of 3-D shapes, including number of edges, vertices and faces.', 2, 0.45, 45),
    ('70102006-0003-0000-0000-000000000001', '00102006-0000-0000-0000-000000000001', 'y2_position_turns', 'Position and Turns',
     'Order and arrange combinations of objects and shapes in patterns and sequences. Use mathematical vocabulary to describe position, direction and movement.', 3, 0.40, 45);

-- Y2 Statistics Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70102007-0001-0000-0000-000000000001', '00102007-0000-0000-0000-000000000001', 'y2_pictograms', 'Pictograms and Tally Charts',
     'Interpret and construct simple pictograms, tally charts, block diagrams and simple tables.', 1, 0.40, 45),
    ('70102007-0002-0000-0000-000000000001', '00102007-0000-0000-0000-000000000001', 'y2_data_questions', 'Asking Questions About Data',
     'Ask and answer simple questions by counting the number of objects in each category and sorting the categories.', 2, 0.45, 45);

-- ============================================================================
-- 5. TOPICS - MATHEMATICS YEAR 3
-- ============================================================================

-- Y3 Number and Place Value Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103001-0001-0000-0000-000000000001', '00103001-0000-0000-0000-000000000001', 'y3_count_multiples', 'Counting in Multiples',
     'Count from 0 in multiples of 4, 8, 50 and 100; find 10 or 100 more or less than a given number.', 1, 0.45, 45),
    ('70103001-0002-0000-0000-000000000001', '00103001-0000-0000-0000-000000000001', 'y3_place_value_1000', 'Place Value to 1000',
     'Recognise the place value of each digit in a three-digit number (hundreds, tens, ones).', 2, 0.45, 60),
    ('70103001-0003-0000-0000-000000000001', '00103001-0000-0000-0000-000000000001', 'y3_compare_order_1000', 'Compare and Order to 1000',
     'Compare and order numbers up to 1000.', 3, 0.45, 45),
    ('70103001-0004-0000-0000-000000000001', '00103001-0000-0000-0000-000000000001', 'y3_roman_numerals_12', 'Roman Numerals I to XII',
     'Read and write numbers to 100 in numerals and in words. Read Roman numerals I to XII (for clock faces).', 4, 0.50, 45);

-- Y3 Addition and Subtraction Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103002-0001-0000-0000-000000000001', '00103002-0000-0000-0000-000000000001', 'y3_mental_3digit', 'Mental Addition and Subtraction',
     'Add and subtract numbers mentally, including: a three-digit number and ones, tens, or hundreds.', 1, 0.50, 60),
    ('70103002-0002-0000-0000-000000000001', '00103002-0000-0000-0000-000000000001', 'y3_columnar_methods', 'Columnar Addition and Subtraction',
     'Add and subtract numbers with up to three digits using formal written methods of columnar addition and subtraction.', 2, 0.55, 60),
    ('70103002-0003-0000-0000-000000000001', '00103002-0000-0000-0000-000000000001', 'y3_estimate_check', 'Estimate and Check',
     'Estimate the answer to a calculation and use inverse operations to check answers.', 3, 0.50, 45),
    ('70103002-0004-0000-0000-000000000001', '00103002-0000-0000-0000-000000000001', 'y3_missing_number_problems', 'Missing Number Problems',
     'Solve problems, including missing number problems, using number facts, place value and more complex addition and subtraction.', 4, 0.55, 60);

-- Y3 Multiplication and Division Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103003-0001-0000-0000-000000000001', '00103003-0000-0000-0000-000000000001', 'y3_3_times_table', '3 Times Table',
     'Recall and use multiplication and division facts for the 3 multiplication table.', 1, 0.50, 45),
    ('70103003-0002-0000-0000-000000000001', '00103003-0000-0000-0000-000000000001', 'y3_4_times_table', '4 Times Table',
     'Recall and use multiplication and division facts for the 4 multiplication table.', 2, 0.50, 45),
    ('70103003-0003-0000-0000-000000000001', '00103003-0000-0000-0000-000000000001', 'y3_8_times_table', '8 Times Table',
     'Recall and use multiplication and division facts for the 8 multiplication table.', 3, 0.55, 45),
    ('70103003-0004-0000-0000-000000000001', '00103003-0000-0000-0000-000000000001', 'y3_multiply_2digit_1digit', 'Multiply 2-Digit by 1-Digit',
     'Write and calculate mathematical statements for multiplication using known tables, progressing to two-digit by one-digit.', 4, 0.55, 60),
    ('70103003-0005-0000-0000-000000000001', '00103003-0000-0000-0000-000000000001', 'y3_scaling_problems', 'Scaling Problems',
     'Solve problems involving multiplication including scaling problems and correspondence problems.', 5, 0.60, 60);

-- Y3 Fractions Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103004-0001-0000-0000-000000000001', '00103004-0000-0000-0000-000000000001', 'y3_tenths', 'Counting in Tenths',
     'Count up and down in tenths; recognise that tenths arise from dividing an object into 10 equal parts.', 1, 0.50, 45),
    ('70103004-0002-0000-0000-000000000001', '00103004-0000-0000-0000-000000000001', 'y3_fractions_as_numbers', 'Fractions as Numbers',
     'Recognise, find and write fractions of a discrete set of objects: unit fractions and non-unit fractions with small denominators.', 2, 0.55, 60),
    ('70103004-0003-0000-0000-000000000001', '00103004-0000-0000-0000-000000000001', 'y3_equivalent_fractions', 'Equivalent Fractions',
     'Recognise and show, using diagrams, equivalent fractions with small denominators.', 3, 0.55, 45),
    ('70103004-0004-0000-0000-000000000001', '00103004-0000-0000-0000-000000000001', 'y3_add_subtract_fractions', 'Add and Subtract Fractions',
     'Add and subtract fractions with the same denominator within one whole.', 4, 0.55, 60),
    ('70103004-0005-0000-0000-000000000001', '00103004-0000-0000-0000-000000000001', 'y3_compare_fractions', 'Compare and Order Fractions',
     'Compare and order unit fractions, and fractions with the same denominators.', 5, 0.55, 45);

-- Y3 Measurement Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103005-0001-0000-0000-000000000001', '00103005-0000-0000-0000-000000000001', 'y3_length', 'Length (m/cm/mm)',
     'Measure, compare, add and subtract: lengths (m/cm/mm).', 1, 0.45, 45),
    ('70103005-0002-0000-0000-000000000001', '00103005-0000-0000-0000-000000000001', 'y3_mass', 'Mass (kg/g)',
     'Measure, compare, add and subtract: mass (kg/g).', 2, 0.45, 45),
    ('70103005-0003-0000-0000-000000000001', '00103005-0000-0000-0000-000000000001', 'y3_capacity', 'Capacity (l/ml)',
     'Measure, compare, add and subtract: volume/capacity (l/ml).', 3, 0.45, 45),
    ('70103005-0004-0000-0000-000000000001', '00103005-0000-0000-0000-000000000001', 'y3_perimeter', 'Perimeter',
     'Measure the perimeter of simple 2-D shapes.', 4, 0.50, 45),
    ('70103005-0005-0000-0000-000000000001', '00103005-0000-0000-0000-000000000001', 'y3_time_analogue', 'Time (Analogue Clocks)',
     'Tell and write the time from an analogue clock, including using Roman numerals from I to XII.', 5, 0.50, 60),
    ('70103005-0006-0000-0000-000000000001', '00103005-0000-0000-0000-000000000001', 'y3_time_duration', 'Time Duration',
     'Estimate and read time with increasing accuracy; record and compare time in terms of seconds, minutes and hours.', 6, 0.55, 45);

-- Y3 Geometry Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103006-0001-0000-0000-000000000001', '00103006-0000-0000-0000-000000000001', 'y3_right_angles', 'Right Angles',
     'Recognise angles as a property of shape or a description of a turn. Identify right angles.', 1, 0.45, 45),
    ('70103006-0002-0000-0000-000000000001', '00103006-0000-0000-0000-000000000001', 'y3_compare_angles', 'Comparing Angles',
     'Identify whether angles are greater than or less than a right angle.', 2, 0.50, 45),
    ('70103006-0003-0000-0000-000000000001', '00103006-0000-0000-0000-000000000001', 'y3_perpendicular_parallel', 'Perpendicular and Parallel Lines',
     'Identify horizontal and vertical lines and pairs of perpendicular and parallel lines.', 3, 0.50, 45),
    ('70103006-0004-0000-0000-000000000001', '00103006-0000-0000-0000-000000000001', 'y3_2d_3d_shapes', '2-D and 3-D Shapes',
     'Draw 2-D shapes and make 3-D shapes using modelling materials; recognise 3-D shapes in different orientations.', 4, 0.50, 60);

-- Y3 Statistics Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70103007-0001-0000-0000-000000000001', '00103007-0000-0000-0000-000000000001', 'y3_bar_charts', 'Bar Charts',
     'Interpret and present data using bar charts.', 1, 0.45, 45),
    ('70103007-0002-0000-0000-000000000001', '00103007-0000-0000-0000-000000000001', 'y3_pictograms', 'Pictograms',
     'Interpret and present data using pictograms where the symbol represents more than one unit.', 2, 0.50, 45),
    ('70103007-0003-0000-0000-000000000001', '00103007-0000-0000-0000-000000000001', 'y3_tables', 'Tables',
     'Interpret and present data using tables. Solve one-step and two-step questions using information presented in tables.', 3, 0.50, 45);

-- ============================================================================
-- 5. TOPICS - MATHEMATICS YEAR 4
-- ============================================================================

-- Y4 Number and Place Value Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104001-0001-0000-0000-000000000001', '00104001-0000-0000-0000-000000000001', 'y4_count_multiples', 'Counting in Multiples',
     'Count in multiples of 6, 7, 9, 25 and 1000.', 1, 0.50, 45),
    ('70104001-0002-0000-0000-000000000001', '00104001-0000-0000-0000-000000000001', 'y4_place_value_10000', 'Place Value to 10,000',
     'Recognise the place value of each digit in a four-digit number (thousands, hundreds, tens, and ones).', 2, 0.50, 60),
    ('70104001-0003-0000-0000-000000000001', '00104001-0000-0000-0000-000000000001', 'y4_rounding', 'Rounding Numbers',
     'Round any number to the nearest 10, 100 or 1000.', 3, 0.50, 45),
    ('70104001-0004-0000-0000-000000000001', '00104001-0000-0000-0000-000000000001', 'y4_negative_numbers', 'Negative Numbers',
     'Count backwards through zero to include negative numbers.', 4, 0.55, 45),
    ('70104001-0005-0000-0000-000000000001', '00104001-0000-0000-0000-000000000001', 'y4_roman_numerals_100', 'Roman Numerals to 100',
     'Read Roman numerals to 100 (I to C) and know that over time the numeral system changed.', 5, 0.55, 45);

-- Y4 Addition and Subtraction Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104002-0001-0000-0000-000000000001', '00104002-0000-0000-0000-000000000001', 'y4_columnar_4digit', 'Columnar Methods (4-Digit)',
     'Add and subtract numbers with up to 4 digits using the formal written methods of columnar addition and subtraction.', 1, 0.55, 60),
    ('70104002-0002-0000-0000-000000000001', '00104002-0000-0000-0000-000000000001', 'y4_estimate_check', 'Estimate and Check',
     'Estimate and use inverse operations to check answers to a calculation.', 2, 0.50, 45),
    ('70104002-0003-0000-0000-000000000001', '00104002-0000-0000-0000-000000000001', 'y4_multi_step_problems', 'Multi-Step Problems',
     'Solve addition and subtraction two-step problems in contexts, deciding which operations and methods to use.', 3, 0.60, 60);

-- Y4 Multiplication and Division Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104003-0001-0000-0000-000000000001', '00104003-0000-0000-0000-000000000001', 'y4_times_tables_12', 'Times Tables to 12×12',
     'Recall multiplication and division facts for multiplication tables up to 12 × 12.', 1, 0.55, 90),
    ('70104003-0002-0000-0000-000000000001', '00104003-0000-0000-0000-000000000001', 'y4_factor_pairs', 'Factor Pairs',
     'Recognise and use factor pairs and commutativity in mental calculations.', 2, 0.55, 45),
    ('70104003-0003-0000-0000-000000000001', '00104003-0000-0000-0000-000000000001', 'y4_formal_multiplication', 'Formal Written Multiplication',
     'Multiply two-digit and three-digit numbers by a one-digit number using formal written layout.', 3, 0.60, 60),
    ('70104003-0004-0000-0000-000000000001', '00104003-0000-0000-0000-000000000001', 'y4_mental_division', 'Division with Remainders',
     'Use place value, known and derived facts to multiply and divide mentally, including dividing by 1 and multiplying by 0.', 4, 0.55, 45);

-- Y4 Fractions and Decimals Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104004-0001-0000-0000-000000000001', '00104004-0000-0000-0000-000000000001', 'y4_equivalent_fractions', 'Equivalent Fractions',
     'Recognise and show, using diagrams, families of common equivalent fractions.', 1, 0.55, 60),
    ('70104004-0002-0000-0000-000000000001', '00104004-0000-0000-0000-000000000001', 'y4_add_subtract_fractions', 'Add and Subtract Fractions',
     'Add and subtract fractions with the same denominator.', 2, 0.55, 45),
    ('70104004-0003-0000-0000-000000000001', '00104004-0000-0000-0000-000000000001', 'y4_hundredths', 'Hundredths',
     'Count up and down in hundredths; recognise that hundredths arise when dividing by 100 or dividing tenths by 10.', 3, 0.55, 45),
    ('70104004-0004-0000-0000-000000000001', '00104004-0000-0000-0000-000000000001', 'y4_decimal_equivalents', 'Decimal Equivalents',
     'Recognise and write decimal equivalents of any number of tenths or hundredths. Know decimal equivalents to 1/4, 1/2, 3/4.', 4, 0.55, 60),
    ('70104004-0005-0000-0000-000000000001', '00104004-0000-0000-0000-000000000001', 'y4_compare_decimals', 'Compare Decimals',
     'Compare numbers with the same number of decimal places up to two decimal places.', 5, 0.55, 45),
    ('70104004-0006-0000-0000-000000000001', '00104004-0000-0000-0000-000000000001', 'y4_divide_by_10_100', 'Divide by 10 and 100',
     'Find the effect of dividing a one- or two-digit number by 10 and 100, identifying the value of digits.', 6, 0.55, 45);

-- Y4 Measurement Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104005-0001-0000-0000-000000000001', '00104005-0000-0000-0000-000000000001', 'y4_convert_units', 'Convert Units',
     'Convert between different units of measure (e.g., kilometre to metre; hour to minute).', 1, 0.55, 60),
    ('70104005-0002-0000-0000-000000000001', '00104005-0000-0000-0000-000000000001', 'y4_perimeter', 'Perimeter',
     'Measure and calculate the perimeter of a rectilinear figure (including squares) in centimetres and metres.', 2, 0.50, 45),
    ('70104005-0003-0000-0000-000000000001', '00104005-0000-0000-0000-000000000001', 'y4_area', 'Area',
     'Find the area of rectilinear shapes by counting squares.', 3, 0.55, 45),
    ('70104005-0004-0000-0000-000000000001', '00104005-0000-0000-0000-000000000001', 'y4_time_12_24_hour', 'Time (12 and 24 Hour)',
     'Read, write and convert time between analogue and digital 12- and 24-hour clocks.', 4, 0.55, 60),
    ('70104005-0005-0000-0000-000000000001', '00104005-0000-0000-0000-000000000001', 'y4_money_problems', 'Money Problems',
     'Solve problems involving converting from hours to minutes; minutes to seconds; years to months; weeks to days.', 5, 0.55, 45);

-- Y4 Geometry Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104006-0001-0000-0000-000000000001', '00104006-0000-0000-0000-000000000001', 'y4_classify_shapes', 'Classify Shapes',
     'Compare and classify geometric shapes, including quadrilaterals and triangles, based on their properties and sizes.', 1, 0.50, 60),
    ('70104006-0002-0000-0000-000000000001', '00104006-0000-0000-0000-000000000001', 'y4_angles', 'Acute and Obtuse Angles',
     'Identify acute and obtuse angles and compare and order angles up to two right angles by size.', 2, 0.55, 45),
    ('70104006-0003-0000-0000-000000000001', '00104006-0000-0000-0000-000000000001', 'y4_symmetry', 'Lines of Symmetry',
     'Identify lines of symmetry in 2-D shapes presented in different orientations.', 3, 0.50, 45),
    ('70104006-0004-0000-0000-000000000001', '00104006-0000-0000-0000-000000000001', 'y4_coordinates', 'Coordinates',
     'Describe positions on a 2-D grid as coordinates in the first quadrant.', 4, 0.55, 60),
    ('70104006-0005-0000-0000-000000000001', '00104006-0000-0000-0000-000000000001', 'y4_translations', 'Translations',
     'Describe movements between positions as translations of a given unit to the left/right and up/down.', 5, 0.55, 45);

-- Y4 Statistics Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70104007-0001-0000-0000-000000000001', '00104007-0000-0000-0000-000000000001', 'y4_bar_charts', 'Bar Charts',
     'Interpret and present discrete and continuous data using appropriate graphical methods, including bar charts.', 1, 0.50, 45),
    ('70104007-0002-0000-0000-000000000001', '00104007-0000-0000-0000-000000000001', 'y4_time_graphs', 'Time Graphs',
     'Interpret and present continuous data using time graphs.', 2, 0.55, 60),
    ('70104007-0003-0000-0000-000000000001', '00104007-0000-0000-0000-000000000001', 'y4_solve_data_problems', 'Solving Data Problems',
     'Solve comparison, sum and difference problems using information presented in bar charts, pictograms, tables.', 3, 0.55, 45);

-- ============================================================================
-- 5. TOPICS - MATHEMATICS YEAR 5
-- ============================================================================

-- Y5 Number and Place Value Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105001-0001-0000-0000-000000000001', '00105001-0000-0000-0000-000000000001', 'y5_place_value_million', 'Place Value to 1,000,000',
     'Read, write, order and compare numbers to at least 1,000,000 and determine the value of each digit.', 1, 0.55, 60),
    ('70105001-0002-0000-0000-000000000001', '00105001-0000-0000-0000-000000000001', 'y5_powers_of_10', 'Powers of 10',
     'Count forwards or backwards in steps of powers of 10 for any given number up to 1,000,000.', 2, 0.55, 45),
    ('70105001-0003-0000-0000-000000000001', '00105001-0000-0000-0000-000000000001', 'y5_negative_numbers', 'Negative Numbers',
     'Interpret negative numbers in context, count forwards and backwards with positive and negative whole numbers through zero.', 3, 0.55, 45),
    ('70105001-0004-0000-0000-000000000001', '00105001-0000-0000-0000-000000000001', 'y5_rounding', 'Rounding',
     'Round any number up to 1,000,000 to the nearest 10, 100, 1000, 10,000 and 100,000.', 4, 0.55, 45),
    ('70105001-0005-0000-0000-000000000001', '00105001-0000-0000-0000-000000000001', 'y5_roman_numerals_1000', 'Roman Numerals to 1000',
     'Read Roman numerals to 1000 (M) and recognise years written in Roman numerals.', 5, 0.55, 45);

-- Y5 Addition and Subtraction Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105002-0001-0000-0000-000000000001', '00105002-0000-0000-0000-000000000001', 'y5_formal_methods', 'Formal Written Methods',
     'Add and subtract whole numbers with more than 4 digits, including using formal written methods (columnar addition and subtraction).', 1, 0.55, 60),
    ('70105002-0002-0000-0000-000000000001', '00105002-0000-0000-0000-000000000001', 'y5_mental_strategies', 'Mental Strategies',
     'Add and subtract numbers mentally with increasingly large numbers.', 2, 0.55, 45),
    ('70105002-0003-0000-0000-000000000001', '00105002-0000-0000-0000-000000000001', 'y5_inverse_rounding', 'Inverse and Rounding',
     'Use rounding to check answers to calculations and determine, in the context of a problem, levels of accuracy.', 3, 0.55, 45),
    ('70105002-0004-0000-0000-000000000001', '00105002-0000-0000-0000-000000000001', 'y5_multi_step', 'Multi-Step Problems',
     'Solve addition and subtraction multi-step problems in contexts, deciding which operations and methods to use.', 4, 0.60, 60);

-- Y5 Multiplication and Division Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105003-0001-0000-0000-000000000001', '00105003-0000-0000-0000-000000000001', 'y5_factors_multiples', 'Factors and Multiples',
     'Identify multiples and factors, including finding all factor pairs of a number, and common factors of two numbers.', 1, 0.55, 60),
    ('70105003-0002-0000-0000-000000000001', '00105003-0000-0000-0000-000000000001', 'y5_prime_numbers', 'Prime Numbers',
     'Know and use the vocabulary of prime numbers, prime factors and composite numbers. Establish whether a number up to 100 is prime.', 2, 0.60, 60),
    ('70105003-0003-0000-0000-000000000001', '00105003-0000-0000-0000-000000000001', 'y5_square_cube', 'Square and Cube Numbers',
     'Recognise and use square numbers and cube numbers, and the notation for squared (²) and cubed (³).', 3, 0.55, 45),
    ('70105003-0004-0000-0000-000000000001', '00105003-0000-0000-0000-000000000001', 'y5_multiply_4digit', 'Multiply up to 4-Digit by 2-Digit',
     'Multiply numbers up to 4 digits by a one or two-digit number using a formal written method.', 4, 0.60, 60),
    ('70105003-0005-0000-0000-000000000001', '00105003-0000-0000-0000-000000000001', 'y5_short_division', 'Short Division',
     'Divide numbers up to 4 digits by a one-digit number using the formal written method of short division.', 5, 0.60, 60),
    ('70105003-0006-0000-0000-000000000001', '00105003-0000-0000-0000-000000000001', 'y5_multiply_divide_10_100_1000', 'Multiply and Divide by 10, 100, 1000',
     'Multiply and divide whole numbers and those involving decimals by 10, 100 and 1000.', 6, 0.55, 45);

-- Y5 Fractions, Decimals, Percentages Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105004-0001-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_compare_fractions', 'Compare and Order Fractions',
     'Compare and order fractions whose denominators are all multiples of the same number.', 1, 0.55, 45),
    ('70105004-0002-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_mixed_improper', 'Mixed Numbers and Improper Fractions',
     'Recognise mixed numbers and improper fractions and convert from one form to the other.', 2, 0.60, 60),
    ('70105004-0003-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_add_subtract_fractions', 'Add and Subtract Fractions',
     'Add and subtract fractions with the same denominator and denominators that are multiples of the same number.', 3, 0.60, 60),
    ('70105004-0004-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_multiply_fractions', 'Multiply Fractions',
     'Multiply proper fractions and mixed numbers by whole numbers, supported by materials and diagrams.', 4, 0.60, 60),
    ('70105004-0005-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_thousandths', 'Thousandths',
     'Read and write decimal numbers as fractions. Recognise and use thousandths and relate them to tenths, hundredths and decimals.', 5, 0.55, 45),
    ('70105004-0006-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_percentages', 'Percentages',
     'Recognise the per cent symbol (%) and understand that per cent relates to number of parts per hundred.', 6, 0.55, 60),
    ('70105004-0007-0000-0000-000000000001', '00105004-0000-0000-0000-000000000001', 'y5_fdp_equivalence', 'FDP Equivalence',
     'Write percentages as a fraction with denominator 100, and as a decimal.', 7, 0.55, 45);

-- Y5 Measurement Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105005-0001-0000-0000-000000000001', '00105005-0000-0000-0000-000000000001', 'y5_convert_metric', 'Convert Metric Units',
     'Convert between different units of metric measure (e.g., km and m; cm and m; cm and mm; g and kg; l and ml).', 1, 0.55, 60),
    ('70105005-0002-0000-0000-000000000001', '00105005-0000-0000-0000-000000000001', 'y5_imperial_units', 'Understand Metric and Imperial',
     'Understand and use approximate equivalences between metric units and common imperial units such as inches, pounds and pints.', 2, 0.55, 45),
    ('70105005-0003-0000-0000-000000000001', '00105005-0000-0000-0000-000000000001', 'y5_perimeter_area', 'Perimeter and Area',
     'Measure and calculate the perimeter of composite rectilinear shapes in cm and m. Calculate and compare the area of rectangles.', 3, 0.60, 60),
    ('70105005-0004-0000-0000-000000000001', '00105005-0000-0000-0000-000000000001', 'y5_volume', 'Estimate Volume',
     'Estimate volume (e.g., using 1 cm³ blocks to build cuboids) and capacity (e.g., using water).', 4, 0.55, 45);

-- Y5 Geometry Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105006-0001-0000-0000-000000000001', '00105006-0000-0000-0000-000000000001', 'y5_3d_from_2d', '3-D Shapes from 2-D Representations',
     'Identify 3-D shapes, including cubes and other cuboids, from 2-D representations.', 1, 0.55, 45),
    ('70105006-0002-0000-0000-000000000001', '00105006-0000-0000-0000-000000000001', 'y5_angles_degrees', 'Angles in Degrees',
     'Know angles are measured in degrees: estimate and compare acute, obtuse and reflex angles.', 2, 0.55, 45),
    ('70105006-0003-0000-0000-000000000001', '00105006-0000-0000-0000-000000000001', 'y5_draw_measure_angles', 'Draw and Measure Angles',
     'Draw given angles, and measure them in degrees (°).', 3, 0.55, 60),
    ('70105006-0004-0000-0000-000000000001', '00105006-0000-0000-0000-000000000001', 'y5_angles_on_line', 'Angles on a Straight Line',
     'Identify angles at a point and one whole turn (360°), angles at a point on a straight line and 1/2 a turn (180°).', 4, 0.55, 45),
    ('70105006-0005-0000-0000-000000000001', '00105006-0000-0000-0000-000000000001', 'y5_regular_irregular', 'Regular and Irregular Polygons',
     'Use the properties of rectangles to deduce related facts. Distinguish between regular and irregular polygons.', 5, 0.55, 45),
    ('70105006-0006-0000-0000-000000000001', '00105006-0000-0000-0000-000000000001', 'y5_reflection_translation', 'Reflection and Translation',
     'Identify, describe and represent the position of a shape following a reflection or translation.', 6, 0.60, 60);

-- Y5 Statistics Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70105007-0001-0000-0000-000000000001', '00105007-0000-0000-0000-000000000001', 'y5_line_graphs', 'Line Graphs',
     'Solve comparison, sum and difference problems using information presented in a line graph.', 1, 0.55, 60),
    ('70105007-0002-0000-0000-000000000001', '00105007-0000-0000-0000-000000000001', 'y5_tables_timetables', 'Tables and Timetables',
     'Complete, read and interpret information in tables, including timetables.', 2, 0.55, 45);

-- ============================================================================
-- 5. TOPICS - MATHEMATICS YEAR 6
-- ============================================================================

-- Y6 Number and Place Value Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106001-0001-0000-0000-000000000001', '00106001-0000-0000-0000-000000000001', 'y6_place_value_10m', 'Place Value to 10,000,000',
     'Read, write, order and compare numbers up to 10,000,000 and determine the value of each digit.', 1, 0.60, 60),
    ('70106001-0002-0000-0000-000000000001', '00106001-0000-0000-0000-000000000001', 'y6_rounding', 'Rounding',
     'Round any whole number to a required degree of accuracy.', 2, 0.55, 45),
    ('70106001-0003-0000-0000-000000000001', '00106001-0000-0000-0000-000000000001', 'y6_negative_numbers', 'Negative Numbers in Context',
     'Use negative numbers in context, and calculate intervals across zero.', 3, 0.60, 45);

-- Y6 Four Operations Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106002-0001-0000-0000-000000000001', '00106002-0000-0000-0000-000000000001', 'y6_long_multiplication', 'Long Multiplication',
     'Multiply multi-digit numbers up to 4 digits by a two-digit whole number using the formal written method of long multiplication.', 1, 0.65, 60),
    ('70106002-0002-0000-0000-000000000001', '00106002-0000-0000-0000-000000000001', 'y6_short_division', 'Short Division',
     'Divide numbers up to 4 digits by a two-digit whole number using the formal written method of short division where appropriate.', 2, 0.65, 60),
    ('70106002-0003-0000-0000-000000000001', '00106002-0000-0000-0000-000000000001', 'y6_long_division', 'Long Division',
     'Divide numbers up to 4 digits by a two-digit whole number using the formal written method of long division.', 3, 0.70, 60),
    ('70106002-0004-0000-0000-000000000001', '00106002-0000-0000-0000-000000000001', 'y6_order_of_operations', 'Order of Operations',
     'Use knowledge of the order of operations to carry out calculations involving the four operations.', 4, 0.65, 60),
    ('70106002-0005-0000-0000-000000000001', '00106002-0000-0000-0000-000000000001', 'y6_common_factors_multiples', 'Common Factors and Multiples',
     'Identify common factors, common multiples and prime numbers.', 5, 0.60, 45),
    ('70106002-0006-0000-0000-000000000001', '00106002-0000-0000-0000-000000000001', 'y6_multi_step_problems', 'Multi-Step Problems',
     'Solve problems involving addition, subtraction, multiplication and division. Use estimation to check answers.', 6, 0.65, 60);

-- Y6 Fractions, Decimals, Percentages Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106003-0001-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_simplify_fractions', 'Simplify Fractions',
     'Use common factors to simplify fractions; use common multiples to express fractions in the same denomination.', 1, 0.60, 45),
    ('70106003-0002-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_compare_order_fractions', 'Compare and Order Fractions',
     'Compare and order fractions, including fractions greater than 1.', 2, 0.60, 45),
    ('70106003-0003-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_add_subtract_fractions', 'Add and Subtract Fractions',
     'Add and subtract fractions with different denominators and mixed numbers, using the concept of equivalent fractions.', 3, 0.65, 60),
    ('70106003-0004-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_multiply_fractions', 'Multiply Fractions',
     'Multiply simple pairs of proper fractions, writing the answer in its simplest form.', 4, 0.65, 45),
    ('70106003-0005-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_divide_fractions', 'Divide Fractions',
     'Divide proper fractions by whole numbers.', 5, 0.65, 45),
    ('70106003-0006-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_decimal_places', 'Decimal Places',
     'Identify the value of each digit in numbers given to three decimal places. Multiply and divide by 10, 100, 1000.', 6, 0.60, 45),
    ('70106003-0007-0000-0000-000000000001', '00106003-0000-0000-0000-000000000001', 'y6_fdp_equivalence', 'FDP Equivalence',
     'Recall and use equivalences between simple fractions, decimals and percentages, including in different contexts.', 7, 0.60, 60);

-- Y6 Ratio and Proportion Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106004-0001-0000-0000-000000000001', '00106004-0000-0000-0000-000000000001', 'y6_ratio', 'Ratio',
     'Solve problems involving the relative sizes of two quantities where missing values can be found using multiplication and division facts.', 1, 0.65, 60),
    ('70106004-0002-0000-0000-000000000001', '00106004-0000-0000-0000-000000000001', 'y6_percentages', 'Percentages of Amounts',
     'Solve problems involving the calculation of percentages and the use of percentages for comparison.', 2, 0.65, 60),
    ('70106004-0003-0000-0000-000000000001', '00106004-0000-0000-0000-000000000001', 'y6_scale_factors', 'Scale Factors',
     'Solve problems involving similar shapes where the scale factor is known or can be found.', 3, 0.65, 45),
    ('70106004-0004-0000-0000-000000000001', '00106004-0000-0000-0000-000000000001', 'y6_unequal_sharing', 'Unequal Sharing',
     'Solve problems involving unequal sharing and grouping using knowledge of fractions and multiples.', 4, 0.65, 60);

-- Y6 Algebra Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106005-0001-0000-0000-000000000001', '00106005-0000-0000-0000-000000000001', 'y6_formulae', 'Simple Formulae',
     'Use simple formulae.', 1, 0.60, 45),
    ('70106005-0002-0000-0000-000000000001', '00106005-0000-0000-0000-000000000001', 'y6_sequences', 'Linear Number Sequences',
     'Generate and describe linear number sequences.', 2, 0.60, 45),
    ('70106005-0003-0000-0000-000000000001', '00106005-0000-0000-0000-000000000001', 'y6_missing_numbers', 'Express Missing Numbers Algebraically',
     'Express missing number problems algebraically.', 3, 0.65, 60),
    ('70106005-0004-0000-0000-000000000001', '00106005-0000-0000-0000-000000000001', 'y6_two_unknowns', 'Equations with Two Unknowns',
     'Find pairs of numbers that satisfy an equation with two unknowns.', 4, 0.70, 60),
    ('70106005-0005-0000-0000-000000000001', '00106005-0000-0000-0000-000000000001', 'y6_enumerate', 'Enumerate Possibilities',
     'Enumerate possibilities of combinations of two variables.', 5, 0.65, 45);

-- Y6 Measurement Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106006-0001-0000-0000-000000000001', '00106006-0000-0000-0000-000000000001', 'y6_convert_units', 'Convert Units',
     'Solve problems involving the calculation and conversion of units of measure, using decimal notation up to three decimal places.', 1, 0.60, 60),
    ('70106006-0002-0000-0000-000000000001', '00106006-0000-0000-0000-000000000001', 'y6_miles_km', 'Miles and Kilometres',
     'Use, read, write and convert between standard units, converting measurements of length from a smaller to larger unit. Convert between miles and km.', 2, 0.60, 45),
    ('70106006-0003-0000-0000-000000000001', '00106006-0000-0000-0000-000000000001', 'y6_area_triangles', 'Area of Triangles and Parallelograms',
     'Recognise that shapes with the same areas can have different perimeters. Calculate the area of parallelograms and triangles.', 3, 0.65, 60),
    ('70106006-0004-0000-0000-000000000001', '00106006-0000-0000-0000-000000000001', 'y6_volume', 'Volume of Cuboids',
     'Calculate, estimate and compare volume of cubes and cuboids using standard units.', 4, 0.65, 60);

-- Y6 Geometry Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106007-0001-0000-0000-000000000001', '00106007-0000-0000-0000-000000000001', 'y6_draw_2d', 'Draw 2-D Shapes',
     'Draw 2-D shapes using given dimensions and angles.', 1, 0.60, 60),
    ('70106007-0002-0000-0000-000000000001', '00106007-0000-0000-0000-000000000001', 'y6_3d_nets', '3-D Shapes and Nets',
     'Recognise, describe and build simple 3-D shapes, including making nets.', 2, 0.60, 60),
    ('70106007-0003-0000-0000-000000000001', '00106007-0000-0000-0000-000000000001', 'y6_classify_shapes', 'Classify Shapes',
     'Compare and classify geometric shapes based on their properties and sizes and find unknown angles in any triangles, quadrilaterals, and regular polygons.', 3, 0.65, 60),
    ('70106007-0004-0000-0000-000000000001', '00106007-0000-0000-0000-000000000001', 'y6_angles_shapes', 'Angles in Shapes',
     'Recognise angles where they meet at a point, are on a straight line, or are vertically opposite, and find missing angles.', 4, 0.65, 45),
    ('70106007-0005-0000-0000-000000000001', '00106007-0000-0000-0000-000000000001', 'y6_circles', 'Circles',
     'Illustrate and name parts of circles, including radius, diameter and circumference and know that the diameter is twice the radius.', 5, 0.55, 45),
    ('70106007-0006-0000-0000-000000000001', '00106007-0000-0000-0000-000000000001', 'y6_coordinates', 'Coordinates in Four Quadrants',
     'Describe positions on the full coordinate grid (all four quadrants).', 6, 0.60, 45);

-- Y6 Statistics Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70106008-0001-0000-0000-000000000001', '00106008-0000-0000-0000-000000000001', 'y6_pie_charts', 'Pie Charts',
     'Interpret and construct pie charts and line graphs and use these to solve problems.', 1, 0.60, 60),
    ('70106008-0002-0000-0000-000000000001', '00106008-0000-0000-0000-000000000001', 'y6_line_graphs', 'Line Graphs',
     'Interpret and construct line graphs and use these to solve problems.', 2, 0.55, 45),
    ('70106008-0003-0000-0000-000000000001', '00106008-0000-0000-0000-000000000001', 'y6_mean', 'The Mean',
     'Calculate and interpret the mean as an average.', 3, 0.60, 45);

-- ============================================================================
-- 5. TOPICS - GEOGRAPHY
-- ============================================================================

-- Y1 Geography Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70201001-0001-0000-0000-000000000001', '00201001-0000-0000-0000-000000000001', 'y1_continents', 'Seven Continents',
     'Name and locate the world''s seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, South America.', 1, 0.35, 45),
    ('70201001-0002-0000-0000-000000000001', '00201001-0000-0000-0000-000000000001', 'y1_oceans', 'Five Oceans',
     'Name and locate the world''s five oceans: Atlantic, Pacific, Indian, Southern, Arctic.', 2, 0.35, 45),
    ('70201002-0001-0000-0000-000000000001', '00201002-0000-0000-0000-000000000001', 'y1_uk_countries', 'Countries of the UK',
     'Name and locate the four countries of the United Kingdom: England, Scotland, Wales, Northern Ireland.', 1, 0.30, 45),
    ('70201002-0002-0000-0000-000000000001', '00201002-0000-0000-0000-000000000001', 'y1_uk_capitals', 'UK Capital Cities',
     'Name the capital cities: London, Edinburgh, Cardiff, Belfast.', 2, 0.35, 30),
    ('70201003-0001-0000-0000-000000000001', '00201003-0000-0000-0000-000000000001', 'y1_weather_seasons', 'Weather and Seasons',
     'Identify seasonal and daily weather patterns in the United Kingdom.', 1, 0.30, 45),
    ('70201003-0002-0000-0000-000000000001', '00201003-0000-0000-0000-000000000001', 'y1_physical_features', 'Physical Features',
     'Use basic geographical vocabulary to refer to physical features: beach, cliff, coast, forest, hill, mountain, sea, ocean, river, valley.', 2, 0.35, 45),
    ('70201003-0003-0000-0000-000000000001', '00201003-0000-0000-0000-000000000001', 'y1_human_features', 'Human Features',
     'Use basic geographical vocabulary to refer to human features: city, town, village, factory, farm, house, office, shop.', 3, 0.35, 45),
    ('70201004-0001-0000-0000-000000000001', '00201004-0000-0000-0000-000000000001', 'y1_maps_globes', 'Using Maps and Globes',
     'Use world maps, atlases and globes to identify the UK and its countries.', 1, 0.40, 45);

-- Y2 Geography Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70202001-0001-0000-0000-000000000001', '00202001-0000-0000-0000-000000000001', 'y2_contrasting_uk', 'Our Local Area',
     'Study the geography of a small area of the United Kingdom.', 1, 0.40, 60),
    ('70202001-0002-0000-0000-000000000001', '00202001-0000-0000-0000-000000000001', 'y2_contrasting_country', 'A Contrasting Non-European Country',
     'Compare a small area in a contrasting non-European country (e.g., Kenya, Jamaica).', 2, 0.45, 60),
    ('70202002-0001-0000-0000-000000000001', '00202002-0000-0000-0000-000000000001', 'y2_equator', 'The Equator',
     'Identify the location of hot and cold areas of the world in relation to the Equator.', 1, 0.40, 45),
    ('70202002-0002-0000-0000-000000000001', '00202002-0000-0000-0000-000000000001', 'y2_poles', 'North and South Poles',
     'Identify and locate the North Pole and South Pole on maps and globes.', 2, 0.40, 45),
    ('70202003-0001-0000-0000-000000000001', '00202003-0000-0000-0000-000000000001', 'y2_school_fieldwork', 'School Grounds Fieldwork',
     'Use simple fieldwork and observational skills to study the geography of the school and its grounds.', 1, 0.35, 60);

-- ============================================================================
-- Y3 Geography Topics (UK National Curriculum KS2 - Lower)
-- ============================================================================

-- Y3 Unit: Europe and Russia (u0203001)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70203001-0001-0000-0000-000000000001', '00203001-0000-0000-0000-000000000001', 'y3_european_countries', 'European Countries',
     'Locate European countries using maps, focusing on major nations such as France, Germany, Spain, Italy, and Poland.', 1, 0.45, 60),
    ('70203001-0002-0000-0000-000000000001', '00203001-0000-0000-0000-000000000001', 'y3_european_capitals', 'European Capital Cities',
     'Identify and locate major European capital cities including Paris, Berlin, Madrid, Rome, and Warsaw.', 2, 0.45, 45),
    ('70203001-0003-0000-0000-000000000001', '00203001-0000-0000-0000-000000000001', 'y3_russia_location', 'Russia: Location and Size',
     'Understand that Russia is the world''s largest country, spanning Europe and Asia.', 3, 0.50, 45),
    ('70203001-0004-0000-0000-000000000001', '00203001-0000-0000-0000-000000000001', 'y3_europe_physical', 'Physical Features of Europe',
     'Identify key physical features of Europe: Alps, Rhine, Danube, Mediterranean Sea.', 4, 0.50, 60);

-- Y3 Unit: Climate Zones and Biomes (u0203002)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70203002-0001-0000-0000-000000000001', '00203002-0000-0000-0000-000000000001', 'y3_climate_zones', 'World Climate Zones',
     'Describe and understand the main climate zones: polar, temperate, tropical, and desert.', 1, 0.50, 60),
    ('70203002-0002-0000-0000-000000000001', '00203002-0000-0000-0000-000000000001', 'y3_biomes_intro', 'Introduction to Biomes',
     'Understand that biomes are large regions with similar climate, plants, and animals.', 2, 0.50, 45),
    ('70203002-0003-0000-0000-000000000001', '00203002-0000-0000-0000-000000000001', 'y3_vegetation_belts', 'Vegetation Belts',
     'Identify vegetation belts: tropical rainforest, desert, grassland, deciduous forest, coniferous forest, tundra.', 3, 0.55, 60);

-- Y3 Unit: UK Regions (u0203003)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70203003-0001-0000-0000-000000000001', '00203003-0000-0000-0000-000000000001', 'y3_uk_counties', 'UK Counties',
     'Name and locate counties of England, Scotland, Wales, and Northern Ireland.', 1, 0.45, 60),
    ('70203003-0002-0000-0000-000000000001', '00203003-0000-0000-0000-000000000001', 'y3_uk_cities', 'Major UK Cities',
     'Locate major cities: Manchester, Birmingham, Leeds, Glasgow, Liverpool, Bristol, Sheffield.', 2, 0.45, 45),
    ('70203003-0003-0000-0000-000000000001', '00203003-0000-0000-0000-000000000001', 'y3_uk_geographical_regions', 'UK Geographical Regions',
     'Identify geographical regions: the Highlands, the Lake District, the Pennines, East Anglia, the Midlands.', 3, 0.50, 60),
    ('70203003-0004-0000-0000-000000000001', '00203003-0000-0000-0000-000000000001', 'y3_uk_topography', 'UK Topographical Features',
     'Identify key topographical features: hills, mountains, coasts, rivers, and how land use has changed over time.', 4, 0.50, 60);

-- Y3 Unit: Map Skills (u0203004)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70203004-0001-0000-0000-000000000001', '00203004-0000-0000-0000-000000000001', 'y3_eight_compass', 'Eight Points of a Compass',
     'Use the eight points of a compass: N, NE, E, SE, S, SW, W, NW to describe location and direction.', 1, 0.45, 45),
    ('70203004-0002-0000-0000-000000000001', '00203004-0000-0000-0000-000000000001', 'y3_four_figure_grid', 'Four-Figure Grid References',
     'Use four-figure grid references to locate features on a map (eastings before northings).', 2, 0.50, 60),
    ('70203004-0003-0000-0000-000000000001', '00203004-0000-0000-0000-000000000001', 'y3_map_symbols', 'Map Symbols and Keys',
     'Understand and use standard map symbols and keys to interpret Ordnance Survey maps.', 3, 0.50, 45),
    ('70203004-0004-0000-0000-000000000001', '00203004-0000-0000-0000-000000000001', 'y3_digital_maps', 'Using Digital Maps',
     'Use digital mapping tools (Google Maps, Digimaps) to locate countries and explore features.', 4, 0.45, 45);

-- ============================================================================
-- Y4 Geography Topics (UK National Curriculum KS2 - Lower)
-- ============================================================================

-- Y4 Unit: North and South America (u0204001)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70204001-0001-0000-0000-000000000001', '00204001-0000-0000-0000-000000000001', 'y4_north_america', 'North American Countries',
     'Locate and identify countries of North America: USA, Canada, Mexico, and Central American nations.', 1, 0.50, 60),
    ('70204001-0002-0000-0000-000000000001', '00204001-0000-0000-0000-000000000001', 'y4_south_america', 'South American Countries',
     'Locate and identify countries of South America: Brazil, Argentina, Peru, Colombia, Chile, and others.', 2, 0.50, 60),
    ('70204001-0003-0000-0000-000000000001', '00204001-0000-0000-0000-000000000001', 'y4_americas_physical', 'Physical Features of the Americas',
     'Identify key physical features: Rocky Mountains, Andes, Amazon River, Great Lakes, Grand Canyon.', 3, 0.55, 60),
    ('70204001-0004-0000-0000-000000000001', '00204001-0000-0000-0000-000000000001', 'y4_americas_environments', 'Environmental Regions',
     'Understand environmental regions of the Americas: rainforest, desert, prairie, tundra, and their characteristics.', 4, 0.55, 60);

-- Y4 Unit: Rivers and the Water Cycle (u0204002)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70204002-0001-0000-0000-000000000001', '00204002-0000-0000-0000-000000000001', 'y4_river_features', 'River Features',
     'Describe key river features: source, mouth, tributary, confluence, meander, floodplain, delta, estuary.', 1, 0.50, 60),
    ('70204002-0002-0000-0000-000000000001', '00204002-0000-0000-0000-000000000001', 'y4_water_cycle', 'The Water Cycle',
     'Understand the water cycle: evaporation, condensation, precipitation, collection, and transpiration.', 2, 0.55, 60),
    ('70204002-0003-0000-0000-000000000001', '00204002-0000-0000-0000-000000000001', 'y4_uk_rivers', 'Rivers of the UK',
     'Identify major UK rivers: Thames, Severn, Trent, Great Ouse, Mersey, and their importance.', 3, 0.50, 45),
    ('70204002-0004-0000-0000-000000000001', '00204002-0000-0000-0000-000000000001', 'y4_river_processes', 'River Processes',
     'Understand how rivers shape the landscape through erosion, transportation, and deposition.', 4, 0.55, 60);

-- Y4 Unit: Settlements and Land Use (u0204003)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70204003-0001-0000-0000-000000000001', '00204003-0000-0000-0000-000000000001', 'y4_settlement_types', 'Types of Settlement',
     'Understand different types of settlement: hamlet, village, town, city, and conurbation.', 1, 0.50, 45),
    ('70204003-0002-0000-0000-000000000001', '00204003-0000-0000-0000-000000000001', 'y4_settlement_patterns', 'Settlement Patterns',
     'Identify settlement patterns: linear, nucleated, dispersed, and reasons for their location.', 2, 0.55, 60),
    ('70204003-0003-0000-0000-000000000001', '00204003-0000-0000-0000-000000000001', 'y4_land_use', 'Land Use',
     'Understand different types of land use: residential, commercial, industrial, agricultural, recreational.', 3, 0.50, 45),
    ('70204003-0004-0000-0000-000000000001', '00204003-0000-0000-0000-000000000001', 'y4_economic_activity', 'Economic Activity',
     'Understand economic activity and trade links, including how goods are produced and transported.', 4, 0.55, 60);

-- Y4 Unit: Ordnance Survey Maps (u0204004)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70204004-0001-0000-0000-000000000001', '00204004-0000-0000-0000-000000000001', 'y4_os_maps_intro', 'Introduction to OS Maps',
     'Understand Ordnance Survey maps, their purpose, and different scales (1:50000, 1:25000).', 1, 0.50, 45),
    ('70204004-0002-0000-0000-000000000001', '00204004-0000-0000-0000-000000000001', 'y4_six_figure_grid', 'Six-Figure Grid References',
     'Use six-figure grid references to pinpoint precise locations on OS maps.', 2, 0.55, 60),
    ('70204004-0003-0000-0000-000000000001', '00204004-0000-0000-0000-000000000001', 'y4_os_symbols', 'OS Map Symbols',
     'Recognise and interpret standard Ordnance Survey symbols for features like churches, post offices, footpaths.', 3, 0.50, 45),
    ('70204004-0004-0000-0000-000000000001', '00204004-0000-0000-0000-000000000001', 'y4_contour_lines', 'Contour Lines',
     'Understand and interpret contour lines to identify hills, valleys, and steep/gentle slopes.', 4, 0.55, 60);

-- ============================================================================
-- Y5 Geography Topics (UK National Curriculum KS2 - Upper)
-- ============================================================================

-- Y5 Unit: Latitude, Longitude and Time Zones (u0205001)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70205001-0001-0000-0000-000000000001', '00205001-0000-0000-0000-000000000001', 'y5_latitude_longitude', 'Latitude and Longitude',
     'Identify the position and significance of latitude (horizontal lines) and longitude (vertical lines) on maps and globes.', 1, 0.55, 60),
    ('70205001-0002-0000-0000-000000000001', '00205001-0000-0000-0000-000000000001', 'y5_equator_hemispheres', 'Equator and Hemispheres',
     'Understand the Equator divides Earth into Northern and Southern Hemispheres.', 2, 0.50, 45),
    ('70205001-0003-0000-0000-000000000001', '00205001-0000-0000-0000-000000000001', 'y5_tropics_circles', 'Tropics and Polar Circles',
     'Identify the Tropics of Cancer and Capricorn, and the Arctic and Antarctic Circles.', 3, 0.55, 45),
    ('70205001-0004-0000-0000-000000000001', '00205001-0000-0000-0000-000000000001', 'y5_time_zones', 'Time Zones and Prime Meridian',
     'Understand time zones, the Prime/Greenwich Meridian (0° longitude), and how day and night occur.', 4, 0.55, 60);

-- Y5 Unit: Mountains, Volcanoes and Earthquakes (u0205002)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70205002-0001-0000-0000-000000000001', '00205002-0000-0000-0000-000000000001', 'y5_mountains', 'Mountains',
     'Describe key aspects of mountains: formation (fold, volcanic, dome), features (peak, ridge, valley, glacier).', 1, 0.55, 60),
    ('70205002-0002-0000-0000-000000000001', '00205002-0000-0000-0000-000000000001', 'y5_mountain_ranges', 'World Mountain Ranges',
     'Locate major mountain ranges: Himalayas, Alps, Andes, Rockies, and understand their significance.', 2, 0.55, 45),
    ('70205002-0003-0000-0000-000000000001', '00205002-0000-0000-0000-000000000001', 'y5_volcanoes', 'Volcanoes',
     'Describe and understand volcanoes: structure (crater, magma chamber, vent), types (shield, composite), and effects.', 3, 0.55, 60),
    ('70205002-0004-0000-0000-000000000001', '00205002-0000-0000-0000-000000000001', 'y5_earthquakes', 'Earthquakes and Tectonic Plates',
     'Understand earthquakes: tectonic plates, plate boundaries, fault lines, and the Richter scale.', 4, 0.60, 60);

-- Y5 Unit: European Region Study (u0205003)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70205003-0001-0000-0000-000000000001', '00205003-0000-0000-0000-000000000001', 'y5_european_region', 'Studying a European Region',
     'Conduct an in-depth study of a region in a European country (e.g., Tuscany, Bavaria, Andalusia).', 1, 0.55, 60),
    ('70205003-0002-0000-0000-000000000001', '00205003-0000-0000-0000-000000000001', 'y5_region_physical', 'Physical Geography of the Region',
     'Describe the physical geography of the studied region: climate, landscape, rivers, natural features.', 2, 0.55, 60),
    ('70205003-0003-0000-0000-000000000001', '00205003-0000-0000-0000-000000000001', 'y5_region_human', 'Human Geography of the Region',
     'Describe the human geography: population, settlements, culture, economic activities, land use.', 3, 0.55, 60),
    ('70205003-0004-0000-0000-000000000001', '00205003-0000-0000-0000-000000000001', 'y5_uk_comparison', 'Comparing with UK Region',
     'Compare and contrast the European region with a region in the UK, identifying similarities and differences.', 4, 0.55, 60);

-- Y5 Unit: Fieldwork Investigation (u0205004)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70205004-0001-0000-0000-000000000001', '00205004-0000-0000-0000-000000000001', 'y5_fieldwork_planning', 'Planning Fieldwork',
     'Plan a geographical investigation: forming questions, selecting methods, identifying equipment needed.', 1, 0.50, 45),
    ('70205004-0002-0000-0000-000000000001', '00205004-0000-0000-0000-000000000001', 'y5_data_collection', 'Data Collection Methods',
     'Use fieldwork methods to collect data: observations, measurements, surveys, photographs, sketches.', 2, 0.55, 60),
    ('70205004-0003-0000-0000-000000000001', '00205004-0000-0000-0000-000000000001', 'y5_data_recording', 'Recording and Presenting Data',
     'Record and present data using sketch maps, plans, graphs, tables, and digital technologies.', 3, 0.55, 60),
    ('70205004-0004-0000-0000-000000000001', '00205004-0000-0000-0000-000000000001', 'y5_drawing_conclusions', 'Drawing Conclusions',
     'Analyse fieldwork data and draw conclusions, evaluating the reliability of findings.', 4, 0.55, 45);

-- ============================================================================
-- Y6 Geography Topics (UK National Curriculum KS2 - Upper)
-- ============================================================================

-- Y6 Unit: Natural Resources and Distribution (u0206001)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70206001-0001-0000-0000-000000000001', '00206001-0000-0000-0000-000000000001', 'y6_energy_resources', 'Energy Resources',
     'Describe the distribution of energy resources: oil, gas, coal, nuclear, and renewable sources worldwide.', 1, 0.60, 60),
    ('70206001-0002-0000-0000-000000000001', '00206001-0000-0000-0000-000000000001', 'y6_food_resources', 'Food Resources',
     'Understand the global distribution of food production and factors affecting food security.', 2, 0.55, 60),
    ('70206001-0003-0000-0000-000000000001', '00206001-0000-0000-0000-000000000001', 'y6_mineral_resources', 'Mineral Resources',
     'Identify the distribution of mineral resources: iron, copper, gold, diamonds, and their uses.', 3, 0.55, 45),
    ('70206001-0004-0000-0000-000000000001', '00206001-0000-0000-0000-000000000001', 'y6_water_resources', 'Water Resources',
     'Understand the distribution of fresh water resources and issues of water scarcity globally.', 4, 0.60, 60);

-- Y6 Unit: American Region Study (u0206002)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70206002-0001-0000-0000-000000000001', '00206002-0000-0000-0000-000000000001', 'y6_american_region', 'Studying an American Region',
     'Conduct an in-depth study of a region in North or South America (e.g., California, Amazon Basin, Patagonia).', 1, 0.60, 60),
    ('70206002-0002-0000-0000-000000000001', '00206002-0000-0000-0000-000000000001', 'y6_america_physical', 'Physical Geography of the Region',
     'Describe the physical geography: climate, biomes, landscape features, natural hazards.', 2, 0.55, 60),
    ('70206002-0003-0000-0000-000000000001', '00206002-0000-0000-0000-000000000001', 'y6_america_human', 'Human Geography of the Region',
     'Describe the human geography: population, indigenous peoples, cities, economic activities.', 3, 0.60, 60),
    ('70206002-0004-0000-0000-000000000001', '00206002-0000-0000-0000-000000000001', 'y6_global_comparison', 'Global Comparisons',
     'Compare the American region with regions studied in the UK and Europe, identifying patterns.', 4, 0.55, 45);

-- Y6 Unit: Global Trade and Economics (u0206003)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70206003-0001-0000-0000-000000000001', '00206003-0000-0000-0000-000000000001', 'y6_trade_links', 'Global Trade Links',
     'Understand how countries trade with each other: imports, exports, trade routes, and trade agreements.', 1, 0.60, 60),
    ('70206003-0002-0000-0000-000000000001', '00206003-0000-0000-0000-000000000001', 'y6_supply_chains', 'Supply Chains',
     'Trace the journey of products from raw materials to consumers, understanding global supply chains.', 2, 0.60, 60),
    ('70206003-0003-0000-0000-000000000001', '00206003-0000-0000-0000-000000000001', 'y6_fair_trade', 'Fair Trade',
     'Understand the concept of fair trade and its impact on producers in developing countries.', 3, 0.55, 45),
    ('70206003-0004-0000-0000-000000000001', '00206003-0000-0000-0000-000000000001', 'y6_economic_inequality', 'Economic Inequality',
     'Understand differences between developed and developing countries, and factors affecting economic development.', 4, 0.60, 60);

-- Y6 Unit: Digital Mapping and Analysis (u0206004)
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70206004-0001-0000-0000-000000000001', '00206004-0000-0000-0000-000000000001', 'y6_digital_mapping', 'Digital Mapping Tools',
     'Use digital mapping tools (Google Earth, GIS) to locate countries and explore geographical features.', 1, 0.55, 60),
    ('70206004-0002-0000-0000-000000000001', '00206004-0000-0000-0000-000000000001', 'y6_satellite_imagery', 'Satellite Imagery',
     'Interpret satellite images to identify physical and human features, land use changes over time.', 2, 0.60, 60),
    ('70206004-0003-0000-0000-000000000001', '00206004-0000-0000-0000-000000000001', 'y6_presenting_data', 'Presenting Geographical Information',
     'Present geographical information in a variety of ways: maps, charts, graphs, written reports, digital presentations.', 3, 0.55, 45),
    ('70206004-0004-0000-0000-000000000001', '00206004-0000-0000-0000-000000000001', 'y6_geographical_enquiry', 'Geographical Enquiry',
     'Conduct independent geographical enquiry using multiple sources and present findings.', 4, 0.60, 60);

-- ============================================================================
-- 5. TOPICS - HISTORY
-- ============================================================================

-- Y1 History Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70301001-0001-0000-0000-000000000001', '00301001-0000-0000-0000-000000000001', 'y1_past_present', 'Past and Present',
     'Develop an awareness of the past using common words and phrases relating to the passing of time.', 1, 0.30, 45),
    ('70301001-0002-0000-0000-000000000001', '00301001-0000-0000-0000-000000000001', 'y1_toys_then_now', 'Toys Then and Now',
     'Learn about changes in toys and games within living memory.', 2, 0.30, 45),
    ('70301002-0001-0000-0000-000000000001', '00301002-0000-0000-0000-000000000001', 'y1_gunpowder_plot', 'The Gunpowder Plot',
     'Learn about Guy Fawkes and the Gunpowder Plot of 1605.', 1, 0.35, 45),
    ('70301003-0001-0000-0000-000000000001', '00301003-0000-0000-0000-000000000001', 'y1_neil_armstrong', 'Neil Armstrong',
     'Learn about Neil Armstrong and the first moon landing.', 1, 0.35, 45),
    ('70301003-0002-0000-0000-000000000001', '00301003-0000-0000-0000-000000000001', 'y1_rosa_parks', 'Rosa Parks',
     'Learn about Rosa Parks and her contribution to civil rights.', 2, 0.35, 45);

-- Y2 History Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70302001-0001-0000-0000-000000000001', '00302001-0000-0000-0000-000000000001', 'y2_gfol_causes', 'Causes of the Great Fire',
     'Learn about the causes of the Great Fire of London in 1666.', 1, 0.40, 45),
    ('70302001-0002-0000-0000-000000000001', '00302001-0000-0000-0000-000000000001', 'y2_gfol_spread', 'Spread and Effects',
     'Learn how the fire spread and its effects on London.', 2, 0.40, 45),
    ('70302001-0003-0000-0000-000000000001', '00302001-0000-0000-0000-000000000001', 'y2_samuel_pepys', 'Samuel Pepys',
     'Learn about Samuel Pepys and his diary as a historical source.', 3, 0.40, 45),
    ('70302002-0001-0000-0000-000000000001', '00302002-0000-0000-0000-000000000001', 'y2_columbus', 'Christopher Columbus',
     'Learn about Christopher Columbus and his voyage to the Americas.', 1, 0.40, 45),
    ('70302002-0002-0000-0000-000000000001', '00302002-0000-0000-0000-000000000001', 'y2_armstrong', 'Neil Armstrong',
     'Learn about Neil Armstrong and compare exploration then and now.', 2, 0.40, 45),
    ('70302003-0001-0000-0000-000000000001', '00302003-0000-0000-0000-000000000001', 'y2_elizabeth_i', 'Elizabeth I',
     'Learn about the life and reign of Elizabeth I.', 1, 0.45, 60),
    ('70302003-0002-0000-0000-000000000001', '00302003-0000-0000-0000-000000000001', 'y2_queen_victoria', 'Queen Victoria',
     'Learn about the life and reign of Queen Victoria.', 2, 0.45, 60),
    ('70302004-0001-0000-0000-000000000001', '00302004-0000-0000-0000-000000000001', 'y2_florence_nightingale', 'Florence Nightingale',
     'Learn about Florence Nightingale and her work in the Crimean War.', 1, 0.40, 45),
    ('70302004-0002-0000-0000-000000000001', '00302004-0000-0000-0000-000000000001', 'y2_mary_seacole', 'Mary Seacole',
     'Learn about Mary Seacole and her contributions to nursing.', 2, 0.40, 45);

-- Y3-6 History Topics
INSERT INTO topics (id, unit_id, code, name, description, sequence, base_difficulty, estimated_minutes) VALUES
    ('70303001-0001-0000-0000-000000000001', '00303001-0000-0000-0000-000000000001', 'y3_hunter_gatherers', 'Hunter-Gatherers',
     'Learn about late Neolithic hunter-gatherers and early farmers.', 1, 0.45, 60),
    ('70303001-0002-0000-0000-000000000001', '00303001-0000-0000-0000-000000000001', 'y3_skara_brae', 'Skara Brae',
     'Learn about the Stone Age settlement of Skara Brae in Orkney.', 2, 0.45, 45),
    ('70303002-0001-0000-0000-000000000001', '00303002-0000-0000-0000-000000000001', 'y3_stonehenge', 'Stonehenge',
     'Learn about Stonehenge and Bronze Age monuments.', 1, 0.50, 60),
    ('70303003-0001-0000-0000-000000000001', '00303003-0000-0000-0000-000000000001', 'y3_iron_age_hillforts', 'Iron Age Hill Forts',
     'Learn about Iron Age hill forts and tribal kingdoms.', 1, 0.50, 60),
    ('70304001-0001-0000-0000-000000000001', '00304001-0000-0000-0000-000000000001', 'y4_julius_caesar', 'Julius Caesar''s Invasions',
     'Learn about Julius Caesar''s attempted invasions of Britain in 55-54 BC.', 1, 0.50, 60),
    ('70304001-0002-0000-0000-000000000001', '00304001-0000-0000-0000-000000000001', 'y4_claudius_invasion', 'The Roman Conquest',
     'Learn about the successful Roman invasion under Emperor Claudius in AD 43.', 2, 0.50, 60),
    ('70304002-0001-0000-0000-000000000001', '00304002-0000-0000-0000-000000000001', 'y4_boudica', 'Boudica''s Rebellion',
     'Learn about Queen Boudica and British resistance to Roman rule.', 1, 0.50, 60),
    ('70304002-0002-0000-0000-000000000001', '00304002-0000-0000-0000-000000000001', 'y4_roman_roads', 'Romanisation of Britain',
     'Learn about Roman roads, towns, Hadrian''s Wall, and daily life in Roman Britain.', 2, 0.50, 60),
    ('70304003-0001-0000-0000-000000000001', '00304003-0000-0000-0000-000000000001', 'y4_pyramids', 'Egyptian Pyramids',
     'Learn about the building of pyramids and their significance.', 1, 0.50, 60),
    ('70304003-0002-0000-0000-000000000001', '00304003-0000-0000-0000-000000000001', 'y4_tutankhamun', 'Tutankhamun',
     'Learn about Tutankhamun and Howard Carter''s discovery.', 2, 0.50, 45),
    ('70305001-0001-0000-0000-000000000001', '00305001-0000-0000-0000-000000000001', 'y5_anglo_saxon_invasion', 'Anglo-Saxon Invasions',
     'Learn about the Anglo-Saxon invasions, settlements and kingdoms.', 1, 0.55, 60),
    ('70305001-0002-0000-0000-000000000001', '00305001-0000-0000-0000-000000000001', 'y5_anglo_saxon_life', 'Anglo-Saxon Life',
     'Learn about Anglo-Saxon village life, art and culture.', 2, 0.50, 60),
    ('70305002-0001-0000-0000-000000000001', '00305002-0000-0000-0000-000000000001', 'y5_viking_raids', 'Viking Raids',
     'Learn about Viking raids and the Danelaw.', 1, 0.55, 60),
    ('70305002-0002-0000-0000-000000000001', '00305002-0000-0000-0000-000000000001', 'y5_alfred_great', 'Alfred the Great',
     'Learn about Alfred the Great and his resistance to Viking invasion.', 2, 0.55, 60),
    ('70305003-0001-0000-0000-000000000001', '00305003-0000-0000-0000-000000000001', 'y5_greek_city_states', 'Greek City States',
     'Learn about Athens, Sparta, and Greek democracy.', 1, 0.55, 60),
    ('70305003-0002-0000-0000-000000000001', '00305003-0000-0000-0000-000000000001', 'y5_greek_achievements', 'Greek Achievements',
     'Learn about Greek achievements: philosophy, Olympics, art, architecture.', 2, 0.55, 60),
    ('70306001-0001-0000-0000-000000000001', '00306001-0000-0000-0000-000000000001', 'y6_1066_claimants', '1066: Claims to the Throne',
     'Learn about the contenders for the throne after Edward the Confessor''s death.', 1, 0.55, 60),
    ('70306001-0002-0000-0000-000000000001', '00306001-0000-0000-0000-000000000001', 'y6_battle_hastings', 'The Battle of Hastings',
     'Learn about the Battle of Hastings and the Norman Conquest.', 2, 0.55, 60),
    ('70306002-0001-0000-0000-000000000001', '00306002-0000-0000-0000-000000000001', 'y6_mayan_society', 'Mayan Society',
     'Learn about Mayan civilization: cities, rulers, religion, and daily life.', 1, 0.55, 60),
    ('70306002-0002-0000-0000-000000000001', '00306002-0000-0000-0000-000000000001', 'y6_mayan_achievements', 'Mayan Achievements',
     'Learn about Mayan writing, mathematics, calendar, and astronomy.', 2, 0.55, 60),
    ('70306003-0001-0000-0000-000000000001', '00306003-0000-0000-0000-000000000001', 'y6_ww2_causes', 'Causes of World War Two',
     'Learn about the causes and outbreak of World War Two.', 1, 0.60, 60),
    ('70306003-0002-0000-0000-000000000001', '00306003-0000-0000-0000-000000000001', 'y6_britain_at_war', 'Britain at War',
     'Learn about the home front: evacuation, the Blitz, rationing.', 2, 0.55, 60),
    ('70306003-0003-0000-0000-000000000001', '00306003-0000-0000-0000-000000000001', 'y6_ww2_end', 'End of the War',
     'Learn about D-Day, VE Day, and the aftermath of war.', 3, 0.55, 60);

-- ============================================================================
-- 6. LEARNING OBJECTIVES - Sample for Year 1 Mathematics
-- ============================================================================

-- Y1 Counting to 100 Objectives
INSERT INTO learning_objectives (id, topic_id, code, objective, bloom_level, sequence, mastery_threshold) VALUES
    ('10101001-0001-0001-0000-000000000001', '70101001-0001-0000-0000-000000000001', 'y1_count_forward',
     'Count forwards to 100 from any given number.', 'remember', 1, 0.80),
    ('10101001-0001-0002-0000-000000000001', '70101001-0001-0000-0000-000000000001', 'y1_count_backward',
     'Count backwards from 100 to any given number.', 'remember', 2, 0.80),
    ('10101001-0001-0003-0000-000000000001', '70101001-0001-0000-0000-000000000001', 'y1_count_across_tens',
     'Count across tens boundaries (e.g., 28, 29, 30, 31).', 'understand', 3, 0.75);

-- Y1 One More/Less Objectives
INSERT INTO learning_objectives (id, topic_id, code, objective, bloom_level, sequence, mastery_threshold) VALUES
    ('10101001-0003-0001-0000-000000000001', '70101001-0003-0000-0000-000000000001', 'y1_identify_one_more',
     'Given a number, identify one more.', 'understand', 1, 0.80),
    ('10101001-0003-0002-0000-000000000001', '70101001-0003-0000-0000-000000000001', 'y1_identify_one_less',
     'Given a number, identify one less.', 'understand', 2, 0.80);

-- Y1 Number Bonds Objectives
INSERT INTO learning_objectives (id, topic_id, code, objective, bloom_level, sequence, mastery_threshold) VALUES
    ('10101002-0001-0001-0000-000000000001', '70101002-0001-0000-0000-000000000001', 'y1_bonds_10',
     'Recall number bonds to 10.', 'remember', 1, 0.85),
    ('10101002-0001-0002-0000-000000000001', '70101002-0001-0000-0000-000000000001', 'y1_bonds_20',
     'Recall number bonds to 20.', 'remember', 2, 0.80),
    ('10101002-0001-0003-0000-000000000001', '70101002-0001-0000-0000-000000000001', 'y1_use_bonds_subtraction',
     'Use number bonds to solve subtraction facts.', 'apply', 3, 0.75);

-- Y1 2D Shapes Objectives
INSERT INTO learning_objectives (id, topic_id, code, objective, bloom_level, sequence, mastery_threshold) VALUES
    ('10101006-0001-0001-0000-000000000001', '70101006-0001-0000-0000-000000000001', 'y1_name_2d_shapes',
     'Recognise and name common 2-D shapes: circle, triangle, square, rectangle.', 'remember', 1, 0.85),
    ('10101006-0001-0002-0000-000000000001', '70101006-0001-0000-0000-000000000001', 'y1_describe_2d_shapes',
     'Describe properties of 2-D shapes using appropriate vocabulary.', 'understand', 2, 0.75);

-- Y1 Geography - Continents Objectives
INSERT INTO learning_objectives (id, topic_id, code, objective, bloom_level, sequence, mastery_threshold) VALUES
    ('10201001-0001-0001-0000-000000000001', '70201001-0001-0000-0000-000000000001', 'y1_name_continents',
     'Name the seven continents of the world.', 'remember', 1, 0.85),
    ('10201001-0001-0002-0000-000000000001', '70201001-0001-0000-0000-000000000001', 'y1_locate_continents',
     'Locate the seven continents on a world map or globe.', 'understand', 2, 0.80);

-- Y1 History - Past and Present Objectives
INSERT INTO learning_objectives (id, topic_id, code, objective, bloom_level, sequence, mastery_threshold) VALUES
    ('10301001-0001-0001-0000-000000000001', '70301001-0001-0000-0000-000000000001', 'y1_time_vocabulary',
     'Use common words and phrases relating to the passing of time: before, after, a long time ago, then, now.', 'remember', 1, 0.80),
    ('10301001-0001-0002-0000-000000000001', '70301001-0001-0000-0000-000000000001', 'y1_sequence_events',
     'Sequence events and objects in chronological order.', 'understand', 2, 0.75);

-- ============================================================================
-- 7. KNOWLEDGE COMPONENTS - Sample for Year 1 Mathematics
-- ============================================================================

-- Number Bonds KCs
INSERT INTO knowledge_components (id, learning_objective_id, code, name, description, component_type, difficulty, sequence) VALUES
    ('b0101002-0001-0001-0001-000000000001', '10101002-0001-0001-0000-000000000001', 'kc_bonds_10_pairs',
     'Number Pairs to 10', 'Know all pairs of numbers that add to make 10 (0+10, 1+9, 2+8, 3+7, 4+6, 5+5).', 'fact', 0.35, 1),
    ('b0101002-0001-0001-0002-000000000001', '10101002-0001-0001-0000-000000000001', 'kc_bonds_10_recall',
     'Rapid Recall of Bonds to 10', 'Automatically recall number bonds to 10 without counting.', 'skill', 0.40, 2),
    ('b0101002-0001-0002-0001-000000000001', '10101002-0001-0002-0000-000000000001', 'kc_bonds_20_pairs',
     'Number Pairs to 20', 'Know pairs of numbers that add to make 20.', 'fact', 0.45, 1),
    ('b0101002-0001-0003-0001-000000000001', '10101002-0001-0003-0000-000000000001', 'kc_inverse_relationship',
     'Addition-Subtraction Inverse', 'Understand that subtraction is the inverse of addition.', 'concept', 0.45, 1);

-- 2D Shapes KCs
INSERT INTO knowledge_components (id, learning_objective_id, code, name, description, component_type, difficulty, sequence) VALUES
    ('b0101006-0001-0001-0001-000000000001', '10101006-0001-0001-0000-000000000001', 'kc_circle_properties',
     'Circle', 'A circle is a round shape with no corners or straight sides.', 'concept', 0.30, 1),
    ('b0101006-0001-0001-0002-000000000001', '10101006-0001-0001-0000-000000000001', 'kc_triangle_properties',
     'Triangle', 'A triangle has 3 sides and 3 corners (vertices).', 'concept', 0.30, 2),
    ('b0101006-0001-0001-0003-000000000001', '10101006-0001-0001-0000-000000000001', 'kc_square_properties',
     'Square', 'A square has 4 equal sides and 4 right angle corners.', 'concept', 0.30, 3),
    ('b0101006-0001-0001-0004-000000000001', '10101006-0001-0001-0000-000000000001', 'kc_rectangle_properties',
     'Rectangle', 'A rectangle has 4 sides with opposite sides equal and 4 right angle corners.', 'concept', 0.35, 4);

-- Geography KCs
INSERT INTO knowledge_components (id, learning_objective_id, code, name, description, component_type, difficulty, sequence) VALUES
    ('b0201001-0001-0001-0001-000000000001', '10201001-0001-0001-0000-000000000001', 'kc_continent_africa',
     'Africa', 'Africa is a continent known for its diverse wildlife, including lions, elephants, and giraffes.', 'fact', 0.35, 1),
    ('b0201001-0001-0001-0002-000000000001', '10201001-0001-0001-0000-000000000001', 'kc_continent_europe',
     'Europe', 'Europe is a continent that includes the United Kingdom, France, Germany, and many other countries.', 'fact', 0.30, 2),
    ('b0201001-0001-0001-0003-000000000001', '10201001-0001-0001-0000-000000000001', 'kc_continent_asia',
     'Asia', 'Asia is the largest continent, home to China, India, Japan, and many other countries.', 'fact', 0.35, 3);

-- History KCs
INSERT INTO knowledge_components (id, learning_objective_id, code, name, description, component_type, difficulty, sequence) VALUES
    ('b0301001-0001-0001-0001-000000000001', '10301001-0001-0001-0000-000000000001', 'kc_word_before',
     'Word: Before', 'Before means happening earlier in time than something else.', 'concept', 0.25, 1),
    ('b0301001-0001-0001-0002-000000000001', '10301001-0001-0001-0000-000000000001', 'kc_word_after',
     'Word: After', 'After means happening later in time than something else.', 'concept', 0.25, 2),
    ('b0301001-0001-0001-0003-000000000001', '10301001-0001-0001-0000-000000000001', 'kc_word_long_ago',
     'Phrase: A Long Time Ago', 'A long time ago refers to events that happened many years in the past.', 'concept', 0.30, 3);

-- ============================================================================
-- 8. PREREQUISITES - Topic-level dependencies
-- ============================================================================

-- Mathematics Prerequisites
INSERT INTO prerequisites (id, source_type, source_id, target_type, target_id, strength) VALUES
    -- Y1 counting is prerequisite for Y1 addition/subtraction
    ('d0000001-0000-0000-0000-000000000001', 'topic', '70101001-0001-0000-0000-000000000001', 'topic', '70101002-0001-0000-0000-000000000001', 1.00),
    -- Y1 addition/subtraction is prerequisite for Y1 multiplication/division
    ('d0000002-0000-0000-0000-000000000001', 'topic', '70101002-0001-0000-0000-000000000001', 'topic', '70101003-0001-0000-0000-000000000001', 0.90),
    -- Y1 halves is prerequisite for Y1 quarters
    ('d0000003-0000-0000-0000-000000000001', 'topic', '70101004-0001-0000-0000-000000000001', 'topic', '70101004-0002-0000-0000-000000000001', 1.00),
    -- Y2 place value depends on Y1 counting
    ('d0000004-0000-0000-0000-000000000001', 'topic', '70101001-0002-0000-0000-000000000001', 'topic', '70102001-0001-0000-0000-000000000001', 0.95),
    -- Y2 times tables depend on Y1 multiplication concepts
    ('d0000005-0000-0000-0000-000000000001', 'topic', '70101003-0001-0000-0000-000000000001', 'topic', '70102003-0001-0000-0000-000000000001', 0.90),
    -- Y3 place value depends on Y2 place value
    ('d0000006-0000-0000-0000-000000000001', 'topic', '70102001-0002-0000-0000-000000000001', 'topic', '70103001-0001-0000-0000-000000000001', 0.95),
    -- Y4 times tables depend on Y3 times tables
    ('d0000007-0000-0000-0000-000000000001', 'topic', '70103003-0001-0000-0000-000000000001', 'topic', '70104003-0001-0000-0000-000000000001', 0.95),
    -- Y5 fractions depend on Y4 equivalent fractions
    ('d0000008-0000-0000-0000-000000000001', 'topic', '70104004-0001-0000-0000-000000000001', 'topic', '70105004-0001-0000-0000-000000000001', 0.90),
    -- Y6 long multiplication depends on Y5 multiplication
    ('d0000009-0000-0000-0000-000000000001', 'topic', '70105003-0001-0000-0000-000000000001', 'topic', '70106002-0001-0000-0000-000000000001', 0.95),
    -- Y6 algebra depends on multiplication understanding
    ('d0000010-0000-0000-0000-000000000001', 'topic', '70105003-0001-0000-0000-000000000001', 'topic', '70106005-0001-0000-0000-000000000001', 0.80);

-- Geography Prerequisites
INSERT INTO prerequisites (id, source_type, source_id, target_type, target_id, strength) VALUES
    -- Y1 continents prerequisite for Y2 comparing places
    ('d0000011-0000-0000-0000-000000000001', 'topic', '70201001-0001-0000-0000-000000000001', 'topic', '70202001-0002-0000-0000-000000000001', 0.85),
    -- Y1 UK countries prerequisite for Y3 UK counties
    ('d0000012-0000-0000-0000-000000000001', 'topic', '70201002-0001-0000-0000-000000000001', 'topic', '70203003-0001-0000-0000-000000000001', 0.90),
    -- Y2 equator/poles prerequisite for Y5 latitude/longitude
    ('d0000013-0000-0000-0000-000000000001', 'topic', '70202002-0001-0000-0000-000000000001', 'topic', '70205001-0001-0000-0000-000000000001', 0.85),
    -- Y3 climate zones prerequisite for Y6 biomes
    ('d0000014-0000-0000-0000-000000000001', 'topic', '70203002-0001-0000-0000-000000000001', 'topic', '70206002-0001-0000-0000-000000000001', 0.80);

-- History Prerequisites
INSERT INTO prerequisites (id, source_type, source_id, target_type, target_id, strength) VALUES
    -- Y3 Stone Age prerequisite for Y3 Bronze Age
    ('d0000015-0000-0000-0000-000000000001', 'topic', '70303001-0001-0000-0000-000000000001', 'topic', '70303002-0001-0000-0000-000000000001', 1.00),
    -- Y3 Bronze Age prerequisite for Y3 Iron Age
    ('d0000016-0000-0000-0000-000000000001', 'topic', '70303002-0001-0000-0000-000000000001', 'topic', '70303003-0001-0000-0000-000000000001', 1.00),
    -- Y3 Iron Age prerequisite for Y4 Roman invasion
    ('d0000017-0000-0000-0000-000000000001', 'topic', '70303003-0001-0000-0000-000000000001', 'topic', '70304001-0001-0000-0000-000000000001', 0.90),
    -- Y4 Roman Britain prerequisite for Y5 Anglo-Saxon
    ('d0000018-0000-0000-0000-000000000001', 'topic', '70304002-0002-0000-0000-000000000001', 'topic', '70305001-0001-0000-0000-000000000001', 0.85),
    -- Y5 Anglo-Saxon prerequisite for Y5 Vikings
    ('d0000019-0000-0000-0000-000000000001', 'topic', '70305001-0001-0000-0000-000000000001', 'topic', '70305002-0001-0000-0000-000000000001', 0.95),
    -- Y5 Vikings prerequisite for Y6 1066
    ('d0000020-0000-0000-0000-000000000001', 'topic', '70305002-0002-0000-0000-000000000001', 'topic', '70306001-0001-0000-0000-000000000001', 0.90);

COMMIT;

-- ============================================================================
-- SUMMARY
-- ============================================================================
-- This seed script creates:
-- - 1 Curriculum: UK National Curriculum - Primary (2014)
-- - 6 Grade Levels: Year 1-6 (KS1 + KS2)
-- - 3 Subjects: Mathematics, Geography, History
-- - 47 Units across all subjects and grades
-- - 100+ Topics with base difficulties
-- - Sample Learning Objectives (Bloom's Taxonomy aligned)
-- - Sample Knowledge Components (concept, skill, fact, procedure types)
-- - 20 Prerequisite relationships for learning progression
--
-- Source: UK Department for Education National Curriculum (2014)
-- https://www.gov.uk/government/collections/national-curriculum
-- ============================================================================
