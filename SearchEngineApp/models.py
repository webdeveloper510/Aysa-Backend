from django.db import models

# Model to keep record of Product Search Count
class ProductSearchTrack(models.Model):
    TAB_CHOICES = [
        ('profit', 'Profit'),
        ('tax', 'Tax'),
        ('ceo-worker', 'CEO Worker'),
    ]

    #visit_date = models.DateField()
    brand_name = models.CharField(max_length=500)
    product_name = models.CharField(max_length=500)
    tab_type = models.CharField(max_length=100, choices=TAB_CHOICES)
    search_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)  # optional
    updated_at = models.DateTimeField(auto_now=True)      # optional

    class Meta:
        unique_together = ('product_name', 'tab_type')  
        # ensures each product+tab combination is unique

    def __str__(self):
        return f"{self.product_name} ({self.tab_type}) - {self.search_count}"

# Model to Track User visit Count
class VistorTrackCountModel(models.Model):
    user_browser_id = models.CharField(max_length=500)
    visit_date = models.DateField()
    visit_count = models.PositiveIntegerField(default=0)

# Model to keep password for store admin password
class AdminAuthenticationModel(models.Model):
    password = models.CharField(max_length = 1000)
    created_at = models.DateTimeField(auto_now_add = True)
    updated_at = models.DateTimeField(auto_now= True)


class ProfitData(models.Model):
    brand = models.CharField(max_length=500, blank=True , null=True)
    product_name = models.CharField(max_length=500, blank=True , null=True)
    product_type = models.CharField(max_length=500, blank=True , null=True)
    category = models.CharField(max_length=500, blank=True , null=True)
    gender = models.CharField(max_length=500, blank=True , null=True)
    year = models.CharField(max_length=500, blank=True , null=True)
    product_url = models.CharField(max_length=2000, blank=True , null=True)
    release_price = models.CharField(max_length=500, blank=True , null=True)
    profit_margin = models.CharField(max_length=500, blank=True , null=True)
    wholesale_price = models.CharField(max_length= 500 , blank=True , null=True)



class TaxDataModel(models.Model):
    company_name = models.CharField(max_length=500, blank=True , null=True)
    Year = models.CharField(max_length=500, blank=True , null=True)
    taxes_paid = models.CharField(max_length=500, blank=True , null=True)
    taxes_avoided = models.CharField(max_length=500, blank=True , null=True)
   

# Create your models here.
class CEOWokrerModel(models.Model):
    company_name = models.CharField(max_length=500 , blank=True , null=True)
    year = models.CharField(max_length=500 , blank=True , null=True)
    ceo_name = models.CharField(max_length=500 , blank=True , null=True)
    ceo_total_compensation = models.CharField(max_length = 500 ,blank=True, null=True)
    worker_salary = models.CharField(max_length = 500 ,blank=True, null=True)


