from rest_framework import permissions
from rest_framework.views import APIView
from rest_framework.response import Response


class UnAuthorizedTestView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self,request):
        return Response({"hello":"world"})


class AuthorizedTestView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        info = self.request.user.username
        return Response({"info": info})





