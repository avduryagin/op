select ACTIVITY_ID,CONTRACTOR_ID,START_DATE,END_DATE,DISTRIBUTED from sf_activity_plan
where ACTIVITY_ID in {activities} and PLAN_ID={plan}
