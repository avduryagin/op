UPDATE sf_activity
   SET        

       START_DATE= null,
       END_DATE= null,
       contractor_id=null

 WHERE ID in {activity_id}
   