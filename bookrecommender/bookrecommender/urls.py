from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('recommender.urls')),  # ✅ Use the actual app name here
    path('admin/', admin.site.urls),
]