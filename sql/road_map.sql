with p as(
with q as(   
SELECT OBJ_ID1,
       o1.entity_type                         AS OBJ_TYPE1,
       OBJ_ID2,
       o2.entity_type                         AS OBJ_TYPE2,
       ROUND (DIST_KM, 2)                     AS DISTANCE,
       DIST_SPEED                             AS AVG_SPEED,
       ROUND (DIST_HOURS * 60 * 60, 0)        AS DISTANCE_TIME,
       SEASON_ID,
       MOBILITY_ID                            MOBILITY_TYPE_ID
  FROM SF_DISTANCE  d
       inner JOIN "EXP#OBJECT" o1 ON o1.id = d.OBJ_ID1
       inner JOIN "EXP#OBJECT" o2 ON o2.id = d.OBJ_ID2)
select * from q
  where q.OBJ_ID1 in {0})
select * from p
 where p.OBJ_ID2 in {0} 