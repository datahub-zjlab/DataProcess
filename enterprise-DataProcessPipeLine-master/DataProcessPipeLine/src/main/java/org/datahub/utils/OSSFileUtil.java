package org.datahub.utils;

import com.aliyun.oss.model.OSSObject;
import lombok.extern.slf4j.Slf4j;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
public class OSSFileUtil {


    public static File downloadOssFile(OSSObject ossObject) {
        InputStream inputStream = ossObject.getObjectContent();
        //把pdf文件保存到本地，然后使用PDFBox读取该临时文件。这样可以避免直接在输入流上进行操作时出现的错误。
        String tempLocalDir = System.getProperty("user.dir")+"/temp/";
        String tempLocalpath=tempLocalDir+ossObject.getKey();
        File tempFile=createTempFile(tempLocalpath);
        FileOutputStream outputStream=null;
        log.info("temp file:"+tempLocalpath);
        try  {
            outputStream = new FileOutputStream(tempFile);
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer, 0, buffer.length)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }catch (Exception e) {
            log.error("download  File error", e);
        }finally {
            try {
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        return tempFile;
    }
    private static File createTempFile(String localFilePath){
        File tempFile =new File(localFilePath);
        if(!tempFile.getParentFile().exists()){
            tempFile.getParentFile().mkdirs();
        }
        try {
            tempFile.createNewFile();
        } catch (IOException e) {
            log.error("create temp  file error", e);
        }
        return tempFile;
    }

    public static String inspectOssPrefix(String originalsKey, String version, String ID, String workflow) {
        String newPath;
        Pattern pattern = Pattern.compile("^(.*?/.*?)/");
        Matcher matcher = pattern.matcher(originalsKey);
        if (matcher.find()) {
            newPath = matcher.group(1);
        } else {
            newPath = "materials/aixueshu/tmp";
        }
        newPath += "/" + version + "/" + workflow + "/" + ID;
        return newPath;
    }

    public static String inspectOssPrefixPaper(String originalsKey, String version, String DOI, String workflow) {
        String newPath;
        Pattern pattern = Pattern.compile("^(.*?/.*?)/");
        Matcher matcher = pattern.matcher(originalsKey);
        if (matcher.find()) {
            newPath = matcher.group(1);
        } else {
            newPath = "scihub_unzip/tmp";
        }
        newPath += "/" + version + "/" + workflow + "/" + DOI;
        return newPath;
    }

    public static String inspectOssPrefix(String originalsKey, String version, String workflow) {
        String newPath;
        Pattern pattern = Pattern.compile("^(.*?/.*?)/");
        Matcher matcher = pattern.matcher(originalsKey);
        if (matcher.find()) {
            newPath = matcher.group(1);
        } else {
            newPath = "materials/aixueshu/tmp";
        }
        newPath += "/" + version + "/" + workflow;
        return newPath;
    }
}
