from django.db import models

# Create your models here.

class ProductModel(models.Model):
    brand = models.CharField(max_length=500 , blank=True , null=True)
    product_name  = models.CharField(max_length=500 , blank=True , null=True)
    Type = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    product_url = models.CharField(max_length=500 , blank=True , null=True)
    profit_price = models.CharField(max_length = 255 ,blank=True, null=True)
    profit_made = models.CharField(max_length=255 , blank=True, null=True)
    profit_margin = models.CharField(max_length = 255,blank=True, null=True)  


class TaxModel(models.Model):
    company_name = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    tax_paid = models.CharField(max_length=500 , blank=True , null=True)
    tax_avoid = models.CharField(max_length = 255 ,blank=True, null=True)


class CEOWokrerModel(models.Model):
    company_name = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    ceo_name = models.CharField(max_length=500 , blank=True , null=True)
    ceo_total_compensation = models.CharField(max_length = 500 ,blank=True, null=True)
    worker_salary = models.CharField(max_length = 500 ,blank=True, null=True)



class Visitor_Track_Count(models.Model):
    visit_day = models.DateField()
    profit_visitor_track_count = models.IntegerField(default=0)
    tax_visitor_track_count = models.IntegerField(default=0)
    ceo_worker_visitor_track_count = models.IntegerField(default=0)



class AdminAuthenticationModel(models.Model):
    password = models.CharField(max_length = 1000)
    created_at = models.DateTimeField(auto_now_add = True)
    updated_at = models.DateTimeField(auto_now= True)
