# XMLDataset 多文件夹支持说明

## 修改内容

### 1. 强制 img_path 和 ann_path 一一对应

现在 `img_path` 和 `ann_path` 必须是**一一对应**的关系：
- 单文件夹：`img_path: /path/to/images`, `ann_path: /path/to/annotations`
- 多文件夹：
  ```yaml
  img_path: ['/path/to/images1', '/path/to/images2']
  ann_path: ['/path/to/annotations1', '/path/to/annotations2']
  ```

### 2. 修改的文件

- `nanodet/data/dataset/base.py`: 传递 `(img_path, ann_path)` 元组给 `get_data_info`
- `nanodet/data/dataset/xml_dataset.py`: 
  - `get_file_list()` 改为接收配对的文件夹列表
  - 返回 `(img_base, ann_base, xml_rel_path, img_rel_path)` 四元组
  - 在 COCO 格式中保存 `img_base_path` 字段用于图片加载
- `nanodet/data/dataset/coco.py`: 
  - `get_train_data()` 支持从 `img_base_path` 加载图片
  - 向后兼容单文件夹模式

### 3. 性能优势

**旧方案（慢）**: 遍历所有图片文件夹查找匹配的图片文件
**新方案（快）**: 直接通过配对关系确定图片路径，无需搜索

## YAML 配置示例

### 单文件夹模式
```yaml
data:
  train:
    name: XMLDataset
    class_names: ['cat', 'dog']
    img_path: /data/train/images
    ann_path: /data/train/annotations
    ...
```

### 多文件夹模式（一一对应）
```yaml
data:
  train:
    name: XMLDataset
    class_names: ['cat', 'dog']
    img_path: ['/data/train/images1', '/data/train/images2']
    ann_path: ['/data/train/annotations1', '/data/train/annotations2']
    ...
```

## 注意事项

1. **列表长度必须相同**: `img_path` 和 `ann_path` 的列表长度必须一致
2. **顺序对应**: 第 N 个图片文件夹对应第 N 个标注文件夹
3. **XML 文件内容**: XML 中的 `<filename>` 字段仍然需要正确设置
4. **相对路径保持**: 如果 XML 在子文件夹中，图片相对路径会保持相同结构

## 错误处理

如果长度不匹配，会抛出明确的错误信息：
```
ValueError: img_paths and ann_paths must have the same length!
Got img_paths: 2, ann_paths: 3.
Please ensure they are one-to-one correspondence in YAML config.
```
